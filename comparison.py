import numpy as np
import stonesoup
import matplotlib.pyplot as plte
from datetime import datetime, timedelta

# --- Imports for your ID-JPDAF ---
from StoneSoupID.stonesoupID.predictor.kalman import KalmanPredictor as ID_KalmanPredictor
from StoneSoupID.stonesoupID.updater.kalman   import KalmanUpdater   as ID_KalmanUpdater
from StoneSoupID.stonesoupID.hypothesiser.probability import PDAHypothesiser as ID_PDAHypo
from StoneSoupID.stonesoupID.dataassociator.probability import JPDA            as ID_JPDA

# --- Imports for the standard JPDAF ---
from stonesoup.predictor.kalman       import KalmanPredictor   as Std_KalmanPredictor
from stonesoup.updater.kalman         import KalmanUpdater     as Std_KalmanUpdater
from stonesoup.hypothesiser.probability import PDAHypothesiser as Std_PDAHypo
from stonesoup.dataassociator.probability import JPDA            as Std_JPDA
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.detection import TrueDetection, Clutter
from stonesoup.models.measurement.linear import LinearGaussian
from scipy.stats import uniform

from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track
from stonesoup.functions import gm_reduce_single


def simulate_scenario(clutter_density, prob_detect, meas_noise_var, start_time, steps=20):
    """
    Returns:
      truths:   [GroundTruthPath, …]  (one per target)
      measurements:  list of sets of Detection  (length == steps)
    """
    # 1) build two ground‐truth tracks just like your tutorial
    transition_model = CombinedLinearGaussianTransitionModel([
        ConstantVelocity(0.005), ConstantVelocity(0.005)
    ])

    # first target
    truth1 = GroundTruthPath([
        GroundTruthState([0,1,0,1], timestamp=start_time)
    ])
    for k in range(1, steps):
        truth1.append(GroundTruthState(
            transition_model.function(truth1[k-1], noise=True,
                                      time_interval=timedelta(seconds=1)),
            timestamp=start_time + timedelta(seconds=k)
        ))

    # second target
    truth2 = GroundTruthPath([
        GroundTruthState([0,1,20,-1], timestamp=start_time)
    ])
    for k in range(1, steps):
        truth2.append(GroundTruthState(
            transition_model.function(truth2[k-1], noise=True,
                                      time_interval=timedelta(seconds=1)),
            timestamp=start_time + timedelta(seconds=k)
        ))

    truths = [truth1, truth2]

    # 2) generate measurements + clutter
    measurement_model = LinearGaussian(
        ndim_state=4, mapping=(0,2),
        noise_covar=np.eye(2)*meas_noise_var
    )

    all_measurements = []
    for k in range(steps):
        tstamp = start_time + timedelta(seconds=k)
        meas_set = set()

        # true detections
        for truth in truths:
            if np.random.rand() <= prob_detect:
                z = measurement_model.function(truth[k], noise=True)
                meas_set.add(TrueDetection(
                    state_vector=z,
                    groundtruth_path=truth,
                    timestamp=tstamp,
                    measurement_model=measurement_model
                ))

        # clutter
        for truth in truths:
            x0, _, y0, _ = truth[k].state_vector.ravel()
            # sample Poisson number of clutters with mean=clutter_density*area
            n_clutter = np.random.poisson(clutter_density * 100)  # tune '100' to your scene size
            for _ in range(n_clutter):
                x = uniform.rvs(x0-10, 20)
                y = uniform.rvs(y0-10, 20)
                meas_set.add(Clutter(
                    np.array([[x],[y]]),
                    timestamp=tstamp,
                    measurement_model=measurement_model
                ))

        all_measurements.append(meas_set)

    return truths, all_measurements


def run_single_trial(seed, clutter_density, prob_detect, meas_noise_var):
    np.random.seed(seed)
    # --- 1) simulate truth + measurements (as in your tutorial) ---
    start_time = datetime.now()
    # ... build truths, all_measurements with this clutter_density, prob_detect, meas_noise_var ...
    # For brevity, assume we return:
    # truths: list of GroundTruthPath, each with .states[t].state_vector
    # all_measurements: list of sets of Detection at each t
    truths, all_measurements = simulate_scenario(clutter_density, prob_detect, meas_noise_var, start_time)

    # --- 2) build two parallel trackers ---
    tm = CombinedLinearGaussianTransitionModel([ConstantVelocity(.005), ConstantVelocity(.005)])
    meas_model = LinearGaussian(ndim_state=4, mapping=(0,2),
                                noise_covar=np.eye(2)*meas_noise_var)

    id_pred = ID_KalmanPredictor(tm)
    id_upd  = ID_KalmanUpdater(meas_model)
    id_assoc = ID_JPDA(hypothesiser=ID_PDAHypo(predictor=id_pred, updater=id_upd,
                                              clutter_spatial_density=clutter_density,
                                              prob_detect=prob_detect))

    std_pred = Std_KalmanPredictor(tm)
    std_upd  = Std_KalmanUpdater(meas_model)
    std_assoc = Std_JPDA(hypothesiser=Std_PDAHypo(predictor=std_pred, updater=std_upd,
                                                  clutter_spatial_density=clutter_density,
                                                  prob_detect=prob_detect))

    priors = [
        GaussianState([[0],[1],[0],[1]], np.diag([1.5,.5,1.5,.5]), timestamp=start_time),
        GaussianState([[0],[1],[20],[-1]], np.diag([1.5,.5,1.5,.5]), timestamp=start_time)
    ]
    id_tracks  = [Track([priors[0]]), Track([priors[1]])]
    std_tracks = [Track([priors[0]]), Track([priors[1]])]

    T = len(all_measurements)
    id_RMSE   = np.zeros((2, T))
    std_RMSE  = np.zeros((2, T))
    id_NEES   = np.zeros((2, T))
    std_NEES  = np.zeros((2, T))
    id_miss   = np.zeros((2, T))  # missed detection count per target
    std_miss  = np.zeros((2, T))
    id_false  = np.zeros((2, T))  # false association count per target
    std_false = np.zeros((2, T))

    for t, meas_set in enumerate(all_measurements):
        time = start_time + timedelta(seconds=t)
        # helper to run one step for a given pipeline
        def step(tracks, assoc, updater, errors, nees, missed, false):
            hyps = assoc.associate(tracks, meas_set, time)
            for i, tr in enumerate(tracks):
                hs = hyps[tr]
                # count misses and false
                missed[i, t] = sum(1 for h in hs if h.is_missed)
                false[i, t]  = sum(1 for h in hs if (not h.is_missed) and h.probability<0.01)
                # fuse
                posts, ws = [], []
                for h in hs:
                    posts.append(h.prediction if h.is_missed else updater.update(h))
                    ws.append(h.probability)
                means = np.stack([p.state_vector.ravel() for p in posts], axis=1)
                covs  = np.stack([p.covar for p in posts], axis=2)
                pm, Pm = gm_reduce_single(means, covs, np.array(ws))
                # attach fused to track
                tr.append(GaussianState(pm.reshape(-1,1), Pm, timestamp=time))
                # compute RMSE & NEES
                true = truths[i].states[t].state_vector.ravel()
                err  = pm - true
                errors[i, t] = np.linalg.norm(err)
                nees[i, t]  = err @ np.linalg.inv(Pm) @ err
        # run ID and STD
        step(id_tracks,  id_assoc, id_upd,  id_RMSE,  id_NEES,  id_miss,  id_false)
        step(std_tracks, std_assoc, std_upd, std_RMSE, std_NEES, std_miss, std_false)

    return id_RMSE, std_RMSE, id_NEES, std_NEES, id_miss, std_miss, id_false, std_false

def monte_carlo(N, clutter, pdets, noises):
    # grids
    results = {}
    for cd in clutter:
      for pd in pdets:
        for nv in noises:
          all_id_RMSE   = []
          all_std_RMSE  = []
          all_id_NEES   = []
          all_std_NEES  = []
          for seed in range(N):
            id_rmse, std_rmse, id_nees, std_nees, \
            id_mi, std_mi, id_fa, std_fa = run_single_trial(seed, cd, pd, nv)
            all_id_RMSE.append(id_rmse)
            all_std_RMSE.append(std_rmse)
            all_id_NEES.append(id_nees)
            all_std_NEES.append(std_nees)
          # stack and compute mean/std over trials
          key = (cd,pd,nv)
          results[key] = {
            'id_RMSE_mean':  np.mean(all_id_RMSE, axis=0),
            'id_RMSE_std':   np.std(all_id_RMSE, axis=0),
            'std_RMSE_mean': np.mean(all_std_RMSE, axis=0),
            'std_RMSE_std':  np.std(all_std_RMSE, axis=0),
            'id_NEES_mean':  np.mean(all_id_NEES, axis=0),
            'std_NEES_mean': np.mean(all_std_NEES, axis=0),
            # you can add miss/false similarly...
          }
    return results

if __name__ == "__main__":
    Ntrials = 10
    clutter_grid = [0.05, 0.125, 0.3]
    pdets_grid   = [0.7, 0.9, 0.99]
    noise_grid   = [0.5, 0.75, 1.5]

    res = monte_carlo(Ntrials, clutter_grid, pdets_grid, noise_grid)

    # Example plot for one parameter set
    cd,pd,nv = clutter_grid[1], pdets_grid[1], noise_grid[1]
    r = res[(cd,pd,nv)]
    t = np.arange(r['id_RMSE_mean'].shape[1])  # time steps

    plt.figure(figsize=(10,6))
    plt.fill_between(t,
                     r['id_RMSE_mean'][0]-r['id_RMSE_std'][0],
                     r['id_RMSE_mean'][0]+r['id_RMSE_std'][0],
                     alpha=0.3, label='ID ±σ')
    plt.plot(t, r['id_RMSE_mean'][0],             '-', label='ID mean')
    plt.plot(t, r['std_RMSE_mean'][0],            '--', label='STD mean')
    plt.xlabel('Time step')
    plt.ylabel('Position RMSE')
    plt.title(f'Comparison at clutter={cd}, Pd={pd}, noise={nv}')
    plt.legend()
    plt.show()
