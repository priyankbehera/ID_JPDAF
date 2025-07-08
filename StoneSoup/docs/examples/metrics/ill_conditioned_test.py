#!/usr/bin/env python3
import numpy as np
from numpy.linalg import inv, cholesky
from scipy.integrate import solve_ivp
from scipy.linalg import qr
import matplotlib.pyplot as plt

#───────────────────────────────────────────────────────────────────────────────
# 0. Problem setup (stiff Van der Pol oscillator)
#───────────────────────────────────────────────────────────────────────────────
alpha = 1e4
t0, T = 0.0, 2.0
x0 = np.array([2.0, 0.0])
P0 = np.eye(2)
Q = np.eye(2)
G = np.array([[0, 0],
              [0, 1]])

# Van der Pol drift and Jacobian
def f(t, x):
    return np.array([
        x[1],
        alpha * (1 - x[0]**2) * x[1] - x[0]
    ])

def J(x):
    return np.array([
        [0, 1],
        [-2 * alpha * x[0] * x[1] - 1,
         alpha * (1 - x[0]**2)]
    ])

#───────────────────────────────────────────────────────────────────────────────
# 1. Augmented ODE: state + covariance
#───────────────────────────────────────────────────────────────────────────────
def mde_aug(t, y):
    x = y[:2]
    P = y[2:].reshape(2,2)
    dx = f(t, x)
    dP = J(x) @ P + P @ J(x).T + G @ Q @ G.T
    return np.concatenate([dx, dP.ravel()])

#───────────────────────────────────────────────────────────────────────────────
# 2. CD-EKF via solve_ivp on the augmented system
#───────────────────────────────────────────────────────────────────────────────
def cd_ekf(zs, ts, H, R):
    xh, P = x0.copy(), P0.copy()
    out = []
    t_prev = t0
    for zk, tk in zip(zs, ts):
        y0 = np.concatenate([xh, P.ravel()])
        sol = solve_ivp(
            fun=mde_aug,
            t_span=(t_prev, tk),
            y0=y0,
            method='Radau',      # switch to a robust stiff solver
            rtol=1e-6,
            atol=1e-8
        )
        if not sol.success:
            raise RuntimeError(f"Radau failed on [{t_prev},{tk}]")
        yk     = sol.y[:, -1]
        x_pred = yk[:2]
        P_pred = yk[2:].reshape(2,2)

        # prevent S from becoming singular
        S = H @ P_pred @ H.T + R + 1e-9*np.eye(H.shape[0])
        K = P_pred @ H.T @ inv(S)

        xh = x_pred + K @ (zk - H @ x_pred)
        P  = P_pred - K @ H @ P_pred
        out.append(xh.copy())
        t_prev = tk

    return np.array(out)

#───────────────────────────────────────────────────────────────────────────────
# 3. SR-CD-EKF: same time-update, then QR sqrt-update
#───────────────────────────────────────────────────────────────────────────────
def sr_cd_ekf(zs, ts, H, R):
    xh, P = x0.copy(), P0.copy()
    out = []
    t_prev = t0
    for zk, tk in zip(zs, ts):
        y0 = np.concatenate([xh, P.ravel()])
        sol = solve_ivp(
            fun=mde_aug,
            t_span=(t_prev, tk),
            y0=y0,
            method='Radau',
            rtol=1e-6,
            atol=1e-8
        )
        if not sol.success:
            raise RuntimeError(f"Radau failed on [{t_prev},{tk}]")
        yk     = sol.y[:, -1]
        x_pred = yk[:2]
        P_pred = yk[2:].reshape(2,2)

        # square-root update
        S_pred = cholesky(P_pred + 1e-12*np.eye(2))
        Rhalf  = cholesky(R      + 1e-12*np.eye(2))
        M = np.vstack([
            np.hstack([Rhalf,           H @ S_pred]),
            np.hstack([np.zeros_like(S_pred), S_pred])
        ])
        _, Rq = qr(M, mode='economic')
        S_new = Rq[2:, 2:]
        K_bar = Rq[:2, 2:]

        innov = zk - H @ x_pred
        xh    = x_pred + K_bar @ np.linalg.solve(Rhalf.T, innov)
        P     = S_new @ S_new.T
        out.append(xh.copy())
        t_prev = tk

    return np.array(out)

#───────────────────────────────────────────────────────────────────────────────
# 4. Fast SDE simulation + measurement generation
#───────────────────────────────────────────────────────────────────────────────
def simulate_vdp_fast(ts, sigma):
    # (a) deterministic propagation on the 2-state system
    sol = solve_ivp(
        fun=lambda t,x: f(t,x),
        t_span=(t0, ts[-1]),
        y0=x0,
        method='Radau',
        rtol=1e-6,
        atol=1e-8,
        t_eval=ts
    )
    x_det = sol.y.T

    # (b) add one Wiener increment per interval
    x_true = x_det.copy()
    dt = np.diff(np.concatenate(([t0], ts)))
    for i, Δ in enumerate(dt):
        x_true[i,1] += np.random.randn() * np.sqrt(Δ)

    # (c) measurements
    H = np.ones((2,2)) / (1 + sigma)
    R = np.eye(2) * sigma**2
    zs = np.array([H @ xi + np.random.randn(2)*sigma for xi in x_true])
    return x_true, zs, H, R

#───────────────────────────────────────────────────────────────────────────────
# 5. ARMSE computation + plotting
#───────────────────────────────────────────────────────────────────────────────
def compute_armse_both(sigma, deltas, n_mc=50):
    arm_cd, arm_sr = [], []
    for Δ in deltas:
        ts = np.arange(t0+Δ, T+1e-9, Δ)
        errs_cd, errs_sr = [], []
        for _ in range(n_mc):
            x_true, zs, H, R = simulate_vdp_fast(ts, sigma)
            errs_cd.append(np.sum((cd_ekf(zs, ts, H, R)   - x_true)**2))
            errs_sr.append(np.sum((sr_cd_ekf(zs, ts, H, R) - x_true)**2))
        K = len(ts)
        arm_cd.append(np.sqrt(np.mean(errs_cd)/(2*K)))
        arm_sr.append(np.sqrt(np.mean(errs_sr)/(2*K)))
    return np.array(arm_cd), np.array(arm_sr)

if __name__ == "__main__":
    sigma_list = [1e-4, 1e-6, 1e-8, 1e-10]
    delta_list = np.arange(0.1, 1.01, 0.1)

    fig, axs = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    for ax, σ in zip(axs.ravel(), sigma_list):
        arm_cd, arm_sr = compute_armse_both(σ, delta_list)
        ax.plot(delta_list, arm_cd,   'k-o',  label='CD-EKF')
        ax.plot(delta_list, arm_sr,   'k--s', label='SR-CD-EKF')
        ax.set_title(r'$\sigma={:.0e}$'.format(σ))
        ax.set_xlabel('Sampling period δ')
        ax.set_ylabel('Accumulated RMSE')
        ax.grid(True)
        ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
