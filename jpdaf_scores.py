#!/usr/bin/env python
# coding: utf-8
"""
Compute accuracy (RMSE) for the standard StoneSoup JPDAF filter
over a grid of clutter/detection-probability/noise settings.
"""

import numpy as np
from datetime import datetime, timedelta
from scipy.stats import uniform

from stonesoup.predictor.kalman       import KalmanPredictor
from stonesoup.updater.kalman         import KalmanUpdater
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.dataassociator.probability import JPDA

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.models.measurement.linear import LinearGaussian

from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.detection   import TrueDetection, Clutter
from stonesoup.types.state       import GaussianState
from stonesoup.types.track       import Track
from stonesoup.functions         import gm_reduce_single

def simulate_scenario(clutter, pd, noise_var, start, steps=20):
    tm = CombinedLinearGaussianTransitionModel([ConstantVelocity(.005), ConstantVelocity(.005)])
    truth1 = GroundTruthPath([GroundTruthState([0,1,0,1], timestamp=start)])
    truth2 = GroundTruthPath([GroundTruthState([0,1,20,-1], timestamp=start)])
    for k in range(1, steps):
        for truth in (truth1, truth2):
            x = tm.function(truth[k-1], noise=True, time_interval=timedelta(seconds=1))
            truth.append(GroundTruthState(x, timestamp=start+timedelta(seconds=k)))
    truths = [truth1, truth2]

    meas_model = LinearGaussian(ndim_state=4, mapping=(0,2), noise_covar=np.eye(2)*noise_var)
    all_meas = []
    for k in range(steps):
        t = start+timedelta(seconds=k)
        meas_set = set()
        for truth in truths:
            if np.random.rand() <= pd:
                z = meas_model.function(truth[k], noise=True)
                meas_set.add(TrueDetection(z, truth, t, meas_model))
        for truth in truths:
            x0,_,y0,_ = truth[k].state_vector.ravel()
            for _ in range(np.random.poisson(clutter*100)):
                x = uniform.rvs(x0-10,20); y = uniform.rvs(y0-10,20)
                meas_set.add(Clutter(np.array([[x],[y]]),t,meas_model))
        all_meas.append(meas_set)
    return truths, all_meas

def run_jpdaf_once(seed, clutter, pd, noise_var):
    np.random.seed(seed)
    start = datetime.now()
    truths, measurements = simulate_scenario(clutter, pd, noise_var, start)

    tm = CombinedLinearGaussianTransitionModel([ConstantVelocity(.005), ConstantVelocity(.005)])
    meas_model = LinearGaussian(ndim_state=4, mapping=(0,2), noise_covar=np.eye(2)*noise_var)
    pred = KalmanPredictor(tm)
    upd  = KalmanUpdater(meas_model)
    hypo = PDAHypothesiser(predictor=pred, updater=upd,
                           clutter_spatial_density=clutter, prob_detect=pd)
    assoc= JPDA(hypothesiser=hypo)

    prior1 = GaussianState([[0],[1],[0],[1]], np.diag([1.5,.5,1.5,.5]), timestamp=start)
    prior2 = GaussianState([[0],[1],[20],[-1]], np.diag([1.5,.5,1.5,.5]), timestamp=start)
    tracks = [Track([prior1]), Track([prior2])]

    T = len(measurements)
    rmse = np.zeros((2,T))
    for t, meas_set in enumerate(measurements):
        time = start+timedelta(seconds=t)
        hyps = assoc.associate(tracks, meas_set, time)
        for i, tr in enumerate(tracks):
            posts, ws = [], []
            for h in hyps[tr]:
                posts.append(h.prediction if h.measurement is None else upd.update(h))
                ws.append(h.probability)
            m = np.stack([p.state_vector.ravel() for p in posts], axis=1)
            C = np.stack([p.covar for p in posts], axis=2)
            pm,_ = gm_reduce_single(m,C,np.array(ws))
            true = truths[i].states[t].state_vector.ravel()
            rmse[i,t] = np.linalg.norm(pm-true)
            tr.append(GaussianState(pm.reshape(-1,1), None, timestamp=time))  # no further use
    return rmse

def main():
    Ntrials = 10
    clutter_grid = [0.05,0.125,0.3]
    p_grid       = [0.7, 0.9,0.99]
    noise_grid   = [0.5,0.75,1.5]

    print("clutter,p_detect,noise,mean_RMSE,std_RMSE")
    for cd in clutter_grid:
        for pd in p_grid:
            for nv in noise_grid:
                all_rmse = []
                for seed in range(Ntrials):
                    rmse = run_jpdaf_once(seed, cd, pd, nv)
                    all_rmse.append(rmse.flatten())
                arr = np.stack(all_rmse,axis=0)
                m = arr.mean()
                s = arr.std()
                print(f"{cd},{pd},{nv},{m:.3f},{s:.3f}")

if __name__=="__main__":
    main()
