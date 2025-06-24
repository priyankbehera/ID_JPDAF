"""
jpdaf_radar_benchmark.py  (Stone Soup ≥ v1.6)
================================================
Pure‑Python Monte‑Carlo benchmark for **vanilla JPDAF** on radar‑style
simulations.  It sweeps across (#targets, clutter density, PD) and saves
RMS, OSPA, SIAP IDC and runtime to a CSV.

Run »  ``python comparison.py --out radar_benchmark.csv``

Requires:  Stone Soup >= 1.6, numpy, pandas, tqdm.
"""
from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# --- Stone Soup -------------------------------------------------------------
from stonesoup.simulator.platform import PlatformDetectionSimulator
from stonesoup.simulator.simple import MultiTargetGroundTruthSimulator
from stonesoup.sensor.radar import RadarRotatingBearingRange
from stonesoup.models.transition.linear import ConstantVelocity
from stonesoup.models.measurement.nonlinear import Cartesian2DToBearing
from stonesoup.models.clutter import ClutterModel
from stonesoup.types.state import GaussianState
from stonesoup.dataassociator.probability import JPDA
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.measures import Mahalanobis, Euclidean
from stonesoup.metricgenerator.ospametric import GOSPAMetric
from stonesoup.metricgenerator.tracktotruthmetrics import SIAPMetrics

from stonesoup.metricgenerator.manager import MultiManager
from stonesoup.dataassociator.tracktotrack import TrackToTruth

###############################################################################
# Helper functions
###############################################################################

def make_groundtruth(n_targets: int, seed: int = 0):
    """Straight‑line ground‑truth paths with diverse headings."""
    rng = np.random.default_rng(seed)
    truths = []
    for _ in range(n_targets):
        heading = rng.uniform(0, 2 * np.pi)
        speed = rng.uniform(22, 32)  # m/s ≈ 50‑70 mph
        vx, vy = speed * np.cos(heading), speed * np.sin(heading)
        state_vec = np.array([
            rng.uniform(-15e3, -10e3), vx,
            rng.uniform(-3e3, 3e3), vy,
        ])
        truths.append(GaussianState(state_vec, np.diag([1, 1, 1, 1]), timestamp=0))
    return truths


def make_sensor(pd: float, clutter_rate: float):
    """Create rotating‑radar sensor + clutter model."""
    sensor = RadarRotatingBearingRange(
        ndim_state=4,
        position_mapping=(0, 2),
        noise_covar=np.diag([
            5.0 ** 2,        # σ_r² (m²)
            (0.012) ** 2,    # σ_θ² (rad²) ≈ 0.7°
            0.2 ** 2,        # σ_ṙ² (m²/s²)
        ]),
        rpm=30,             # 2 s between scans
        max_range=40_000.0,
        pd=pd,
    )
    clutter = ClutterModel(rate=clutter_rate, coverage=(80_000.0, 2*np.pi))
    return sensor, clutter


def build_simulator(n_targets: int, pd: float, clutter: float, seed: int):
    """Return detection simulator and truth manager."""
    truths = make_groundtruth(n_targets, seed)
    transition = ConstantVelocity(0.2)  # q (m/s²)
    truth_sim = MultiTargetGroundTruthSimulator(
        transition_model=transition,
        initial_state_truths=truths,
        timestep=2.0,     # 2 s
        seed=seed,
        max_duration=90,  # 3 min ⇒ 45 scans
    )
    sensor, clutter_model = make_sensor(pd, clutter)
    measurement_model = RadarRangeBearingRange(ndim_state=4, mapping=(0, 2))
    sim = PlatformDetectionSimulator(
        groundtruth=truth_sim,
        sensors=[sensor],
        clutter=clutter_model,
        measurement_model=measurement_model,
    )
    return sim, truth_sim, sensor, transition


def build_tracker(sensor, predictor, distance_threshold=30.0):
    hypot = DistanceHypothesiser(
        predictor=predictor,
        measurement_model=sensor.measurement_model,
        distance_measure=Mahalanobis(),
        distance_threshold=distance_threshold,
    )
    hypot = PDAHypothesiser(hypot, prob_detect=sensor.pd)
    updater = KalmanUpdater(measurement_model=sensor.measurement_model)
    return JPDA(predictor=predictor, updater=updater, hypothesiser=hypot)


def build_metric_manager():
    pos_meas = Euclidean((0, 2))
    vel_meas = Euclidean((1, 3))
    gens = [
        GOSPAMetric(c=100, p=2, generator_name="gospa", tracks_key="tracks", truths_key="truths"),
        SIAPMetrics(position_measure=pos_meas, velocity_measure=vel_meas,
                    generator_name="siap", tracks_key="tracks", truths_key="truths"),
    ]
    associator = TrackToTruth(association_threshold=30)
    return MultiManager(gens, associator)


def run_once(seed: int, n_targets: int, clutter: float, pd: float) -> Dict[str, float]:
    sim, truth_sim, sensor, transition = build_simulator(n_targets, pd, clutter, seed)
    predictor = transition  # same model
    tracker = build_tracker(sensor, predictor)
    metric_manager = build_metric_manager()

    # --- run scan loop -----------------------------------------------------
    start = time.perf_counter()
    for time_step, detections in sim:
        tracker.step(detections, time=time_step)
    runtime = time.perf_counter() - start

    # --- feed data to metrics --------------------------------------------
    metric_manager.add_data({
        "truths": truth_sim.groundtruth_paths,
        "tracks": tracker.tracks,
    })
    metrics = metric_manager.generate_metrics()

    # Extract scalar means we care about
    rms = metrics["err"]["RMS Error"].value.mean()
    ospa = metrics["ospa"]["OSPA"].value.mean()
    idc = metrics["siap"]["SIAP IDC"].value

    return dict(seed=seed, n_targets=n_targets, clutter=clutter, pd=pd,
                rms=rms, ospa=ospa, idc=idc, runtime=runtime)

###############################################################################
# Entry‑point
###############################################################################

def main():
    ap = argparse.ArgumentParser("JPDAF radar benchmark (Stone Soup)")
    ap.add_argument("--out", type=Path, default=Path("radar_benchmark.csv"),
                    help="Output CSV path [radar_benchmark.csv]")
    ap.add_argument("--trials", type=int, default=50, help="#seeds per cell [50]")
    args = ap.parse_args()

    grid = itertools.product(
        range(args.trials),          # seeds
        [2, 5, 10],                 # #targets
        [0.0, 0.3, 0.6],            # clutter λC / km²/scan
        [0.6, 0.8, 0.95],           # PD
    )

    rows: List[Dict[str, float]] = []
    for seed, nT, lamC, PD in tqdm(list(grid), desc="JPDAF runs"):
        rows.append(run_once(seed, nT, lamC, PD))

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"Saved → {args.out.resolve()}")


if __name__ == "__main__":
    main()
