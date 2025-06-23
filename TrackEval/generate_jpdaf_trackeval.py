#!/usr/bin/env python3
import os, sys

# ──────────────────────────────────────────────────────────────────────────────
# 1) Ensure your local stonesoup folder (TrackEval/stonesoup) is first on sys.path
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_ROOT = os.path.dirname(os.path.realpath(__file__))
# Insert the TrackEval root
sys.path.insert(0, SCRIPT_ROOT)
# Now when you 'import stonesoup' Python will load ./stonesoup, not the pip one.
# ──────────────────────────────────────────────────────────────────────────────

import csv
import numpy as np

# Stone Soup core components
from stonesoup.types.detection import Detection
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.dataassociator.probability import JPDA
from stonesoup.deleter.time import UpdateTimeStepsDeleter
from stonesoup.initiator.simple import MultiMeasurementInitiator
from stonesoup.tracker.simple import MultiTargetTracker

from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    LinearGaussianTransitionModel
)
from stonesoup.models.measurement.linear import LinearGaussian

# === USER CONFIGURATION ===
TRACKEVAL_ROOT = SCRIPT_ROOT
SPLIT         = "train"    # or "test"
DETECTOR      = "FRCNN"    # DPM, FRCNN or SDP
SEQUENCES     = [
    "MOT17-02","MOT17-04","MOT17-05",
    "MOT17-09","MOT17-10","MOT17-11","MOT17-13"
]

GT_BASE    = os.path.join(TRACKEVAL_ROOT, f"data/gt/MOT17/{SPLIT}")
DETS_BASE  = os.path.join(GT_BASE, "{seq}-{det}/det/det.txt")
OUT_FOLDER = os.path.join(TRACKEVAL_ROOT, "data/trackers/mot_challenge/JPDAF")
os.makedirs(OUT_FOLDER, exist_ok=True)

# === Stone Soup Tracker Setup ===
# Build 4 independent 1-D transition sub-models
model_list = [
    LinearGaussianTransitionModel(
        ndim_state=1,
        noise_diffusion_rate=1.0
    )
    for _ in range(4)
]
transition_model = CombinedLinearGaussianTransitionModel(
    model_list=model_list,
    noise_diffusion_rate=1.0
)

# Measurement model: we observe [x, y] from 4-D state [x, vx, y, vy]
measurement_model = LinearGaussian(
    ndim_state=4,
    mapping=[0, 2],
    noise_covar=np.diag([5.0, 5.0])
)

predictor = KalmanPredictor(transition_model)
updater   = KalmanUpdater(measurement_model)

# PDA hypothesiser + JPDA associator
Pd = 0.9
clutter_rate = 1e-4
hypo  = PDAHypothesiser(
    predictor=predictor,
    updater=updater,
    clutter_spatial_density=clutter_rate,
    prob_detect=Pd
)
assoc = JPDA(hypothesiser=hypo)

# Track initiation & deletion
deleter   = UpdateTimeStepsDeleter(time_steps_since_update=3)
initiator = MultiMeasurementInitiator(
    prior_state=None,
    measurement_model=measurement_model,
    deleter=deleter,
    data_associator=assoc,
    updater=updater,
    min_points=1,
    updates_only=True
)

tracker = MultiTargetTracker(
    initiator=initiator,
    deleter=deleter,
    data_associator=assoc,
    predictor=predictor,
    updater=updater
)

# === Helper: parse a det.txt into per-frame Detection lists ===
def load_detections(det_file_path):
    frames = {}
    with open(det_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            frame_num = int(row[0])
            left, top, w, h, conf = map(float, row[2:7])
            cx = left + w/2
            cy = top  + h/2
            det = Detection(
                state_vector=np.array([[cx], [cy]]),
                timestamp=frame_num,
                measurement_model=measurement_model
            )
            det.confidence = conf
            frames.setdefault(frame_num, []).append(det)
    return frames

# === Main loop: run JPDAF and write results ===
for seq in SEQUENCES:
    det_file = DETS_BASE.format(seq=seq, det=DETECTOR)
    detections_by_frame = load_detections(det_file)

    out_path = os.path.join(OUT_FOLDER, f"{seq}-{DETECTOR}.txt")
    with open(out_path, "w") as fh:
        for frame in sorted(detections_by_frame):
            dets = set(detections_by_frame[frame])
            tracks = tracker.predict_update(dets)
            for track in tracks:
                if not track.confirmed:
                    continue
                x, vx, y, vy = track.state.state_vector.ravel()
                width, height = 1.0, 1.0   # adjust to your object dims
                left = x - width/2
                top  = y - height/2
                conf = getattr(track, "confidence", 1.0)
                fh.write(
                    f"{frame},{track.id},"
                    f"{left:.2f},{top:.2f},"
                    f"{width:.2f},{height:.2f},"
                    f"{conf:.2f},-1,-1,-1\n"
                )
    print(f"▶ Wrote {out_path}")

print("✅ All JPDAF files generated in:", OUT_FOLDER)
