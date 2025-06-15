import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from stonesoup.predictor.kalman       import KalmanPredictor
from stonesoup.updater.kalman         import KalmanUpdater
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.dataassociator.probability import JPDA

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.models.measurement.linear import LinearGaussian

from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.detection   import TrueDetection, Clutter

from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track
from stonesoup.functions    import gm_reduce_single

from scipy.stats import uniform

def simulate_scenario(clutter_density, prob_detect, meas_noise_var, start_time, steps=20):
    transition_model = CombinedLinearGaussianTransitionModel([
        ConstantVelocity(0.005), ConstantVelocity(0.005)
    ])

    # build two truth tracks
    truth1 = GroundTruthPath([GroundTruthState([0,1,0,1], timestamp=start_time)])
    truth2 = GroundTruthPath([GroundTruthState([0,1,20,-1], timestamp=start_time)])
    for k in range(1, steps):
        t1 = transition_model.function(truth1[k-1], noise=True,  time_interval=timedelta(seconds=1))
        t2 = transition_model.function(truth2[k-1], noise=True,  time_interval=timedelta(seconds=1))
        truth1.append(GroundTruthState(t1, timestamp=start_time+timedelta(seconds=k)))
        truth2.append(GroundTruthState(t2, timestamp=start_time+timedelta(seconds=k)))

    truths = [truth1, truth2]

    meas_model = LinearGaussian(ndim_state=4, mapping=(0,2),
                                noise_covar=np.eye(2)*meas_noise_var)

    all_measurements = []
    for k in range(steps):
        tstamp = start_time + timedelta(seconds=k)
        meas_set = set()
        for truth in truths:
            if np.random.rand() <= prob_detect:
                z = meas_model.function(truth[k], noise=True)
                meas_set.add(TrueDetection(
                    state_vector=z,
                    groundtruth_path=truth,
                    timestamp=tstamp,
                    measurement_model=meas_model
                ))
        for truth in truths:
            x0, _, y0, _ = truth[k].state_vector.ravel()
            n_clutter = np.random.poisson(clutter_density * 100)
            for _ in range(n_clutter):
                x = uniform.rvs(x0-10, 20)
                y = uniform.rvs(y0-10, 20)
                meas_set.add(Clutter(
                    np.array([[x],[y]]),
                    timestamp=tstamp,
                    measurement_model=meas_model
                ))
        all_measurements.append(meas_set)

    return truths, all_measurements

def run_single_trial(seed, clutter_density, prob_detect, meas_noise_var):
    np.random.seed(seed)
    start_time = datetime.now()
    truths, all_measurements = simulate_scenario(clutter_density, prob_detect, meas_noise_var, start_time)

    tm = CombinedLinearGaussianTransitionModel([ConstantVelocity(.005), ConstantVelocity(.005)])
    meas_model = LinearGaussian(ndim_state=4, mapping=(0,2), noise_covar=np.eye(2)*meas_noise_var)

    std_pred = KalmanPredictor(tm)
    std_upd  = KalmanUpdater(meas_model)
    std_assoc = JPDA(hypothesiser=PDAHypothesiser(
        predictor=std_pred, updater=std_upd,
        clutter_spatial_density=clutter_density, prob_detect=prob_detect
    ))

    priors = [
        GaussianState([[0],[1],[0],[1]],     np.diag([1.5,.5,1.5,.5]), timestamp=start_time),
        GaussianState([[0],[1],[20],[-1]],    np.diag([1.5,.5,1.5,.5]), timestamp=start_time)
    ]
    std_tracks = [Track([priors[0]]), Track([priors[1]])]

    T = len(all_measurements)
    std_RMSE = np.zeros((2, T))
    std_NEES = np.zeros((2, T))
    std_miss = np.zeros((2, T))
    std_false= np.zeros((2, T))

    for t, meas_set in enumerate(all_measurements):
        time = start_time + timedelta(seconds=t)

        hyps = std_assoc.associate(std_tracks, meas_set, time)
        for i, tr in enumerate(std_tracks):
            hs = hyps[tr]
            std_miss[i, t] = sum(1 for h in hs if h.measurement is None)
            std_false[i, t] = sum(1 for h in hs if h.measurement is not None and h.probability < 0.01)


            posts, ws = [], []
            for h in hs:
                if h.measurement is None or h.measurement_prediction is None:
                    posts.append(h.prediction)
                else:
                    posts.append(std_upd.update(h))
                ws.append(h.probability)


            means = np.stack([p.state_vector.ravel() for p in posts], axis=1)
            covs  = np.stack([p.covar for p in posts], axis=2)
            pm, Pm = gm_reduce_single(means, covs, np.array(ws))

            tr.append(GaussianState(pm.reshape(-1,1), Pm, timestamp=time))

            true = truths[i].states[t].state_vector.ravel()
            err  = pm - true
            std_RMSE[i, t] = np.linalg.norm(err)
            std_NEES[i, t] = (err @ np.linalg.inv(Pm) @ err).item()

    return std_RMSE, std_NEES, std_miss, std_false


def monte_carlo(N, clutter, pdets, noises):
    results = {}
    for cd in clutter:
        for pd in pdets:
            for nv in noises:
                all_std_RMSE = []
                all_std_NEES = []
                for seed in range(N):
                    std_rmse, std_nees, _, _ = run_single_trial(seed, cd, pd, nv)
                    all_std_RMSE.append(std_rmse)
                    all_std_NEES.append(std_nees)
                key = (cd, pd, nv)
                results[key] = {
                    'std_RMSE_mean': np.mean(all_std_RMSE, axis=0),
                    'std_RMSE_std':  np.std(all_std_RMSE, axis=0),
                    'std_NEES_mean': np.mean(all_std_NEES, axis=0),
                }
    return results


if __name__ == "__main__":
    Ntrials      = 10
    clutter_grid = [0.05, 0.125, 0.3]
    pdets_grid   = [0.7, 0.9, 0.99]
    noise_grid   = [0.5, 0.75, 1.5]

    res = monte_carlo(Ntrials, clutter_grid, pdets_grid, noise_grid)

    # plot one example
    cd, pd, nv = clutter_grid[1], pdets_grid[1], noise_grid[1]
    r = res[(cd,pd,nv)]
    t = np.arange(r['std_RMSE_mean'].shape[1])

    plt.figure(figsize=(10,6))
    plt.fill_between(t,
                     r['std_RMSE_mean'][0]-r['std_RMSE_std'][0],
                     r['std_RMSE_mean'][0]+r['std_RMSE_std'][0],
                     alpha=0.3, label='STD ±σ')
    plt.plot(t, r['std_RMSE_mean'][0], '--', label='STD mean')
    plt.xlabel('Time step')
    plt.ylabel('Position RMSE')
    plt.title(f'Standard JPDAF at clutter={cd}, Pd={pd}, noise={nv}')
    plt.legend()
    plt.show()
