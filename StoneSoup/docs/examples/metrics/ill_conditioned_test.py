import numpy as np
from numpy.linalg import cholesky, qr, inv
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#───────────────────────────────────────────────────────────────────────────────
# 1. Problem setup (stiff Van der Pol oscillator)
#───────────────────────────────────────────────────────────────────────────────
alpha = 1e4          # stiffness
t0, T = 0.0, 2.0     # simulation window
x0 = np.array([2.0, 0.0])
xhat0 = x0.copy()
P0 = np.eye(2) 
Q = np.eye(2)
G = np.array([[0, 0],
              [0, 1]])

def f(t, x):
    return np.array([
        x[1],
        alpha*(1 - x[0]**2)*x[1] - x[0]
    ])

def J(x):
    return np.array([
        [0, 1],
        [-2*alpha*x[0]*x[1] - 1,
         alpha*(1 - x[0]**2)]
    ])

#───────────────────────────────────────────────────────────────────────────────
# 2. CD-EKF time-update ODEs
#───────────────────────────────────────────────────────────────────────────────
def mde(t, y):
    xh = y[:2]
    P  = y[2:].reshape(2,2)
    dxh = f(t, xh)
    dP  = J(xh) @ P + P @ J(xh).T + G @ Q @ G.T
    return np.hstack((dxh, dP.ravel()))

def cd_ekf(zs, ts, H, R):
    xh, P = xhat0.copy(), P0.copy()
    out = []
    t_prev = t0
    for k, tk in enumerate(ts):
        y0  = np.hstack((xh, P.ravel()))
        sol = solve_ivp(mde, (t_prev, tk), y0,
                        method='BDF', rtol=1e-8, atol=1e-8)
        yk     = sol.y[:, -1]
        x_pred = yk[:2]
        P_pred = yk[2:].reshape(2,2)

        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ inv(S)
        xh = x_pred + K @ (zs[k] - H @ x_pred)
        P  = P_pred - K @ H @ P_pred

        out.append(xh.copy())
        t_prev = tk
    return np.array(out)

#───────────────────────────────────────────────────────────────────────────────
# 3. SR-CD-EKF: square-root measurement update only
#───────────────────────────────────────────────────────────────────────────────
def sr_mde(t, y):
    xh = y[:2]
    S  = y[2:].reshape(2,2)
    dxh = f(t, xh)
    A   = inv(S) @ J(xh) @ S
    B   = inv(S) @ G @ np.sqrt(Q)
    L   = np.tril(A)
    L[np.diag_indices(2)] *= 0.5
    dS  = S @ (L + L.T + B @ B.T)
    return np.hstack((dxh, dS.ravel()))

def sr_cd_ekf(zs, ts, H, R):
    xh, P = xhat0.copy(), P0.copy()
    out = []
    t_prev = t0
    for k, tk in enumerate(ts):
        # — time-update via sr_mde (not mde!) —
        y0  = np.hstack((xh, cholesky(P).ravel()))
        sol = solve_ivp(sr_mde, (t_prev, tk), y0,
                        method='BDF', rtol=1e-9, atol=1e-9)
        yk     = sol.y[:, -1]
        x_pred = yk[:2]
        S_pred = yk[2:].reshape(2,2)

        # QR-based sqrt measurement update
        Rhalf = cholesky(R)
        M = np.vstack([
            np.hstack([Rhalf,          H @ S_pred]),
            np.hstack([np.zeros_like(S_pred), S_pred])
        ])
        _, Rq = qr(M, mode='reduced')
        S_new = Rq[2:, 2:]
        K_bar = Rq[:2, 2:]

        innov = zs[k] - H @ x_pred
        xh    = x_pred + K_bar @ inv(Rhalf).T @ innov
        P     = S_new @ S_new.T

        out.append(xh.copy())
        t_prev = tk
    return np.array(out)

#───────────────────────────────────────────────────────────────────────────────
# 4. True-state + measurements
#───────────────────────────────────────────────────────────────────────────────
def simulate_true_and_measure(ts, sigma):
    sol = solve_ivp(lambda t,x: f(t,x),
                    (t0, T), x0, method='BDF',
                    t_eval=ts, rtol=1e-8, atol=1e-8)
    x_det = sol.y.T

    x_true = []
    for k, xk in enumerate(x_det):
        dt = ts[k] - (ts[k-1] if k>0 else t0)
        w  = np.random.randn(2)*np.sqrt(dt)
        x_true.append(xk + G @ w)
    x_true = np.array(x_true)

    H = np.ones((2,2)) / (1 + sigma)
    R = np.eye(2) * sigma**2
    zs = np.array([H @ xk + np.random.randn(2)*sigma for xk in x_true])
    return x_true, zs

#───────────────────────────────────────────────────────────────────────────────
# 5. ARMSE + plotting
#───────────────────────────────────────────────────────────────────────────────
def compute_armse(sigma, deltas, n_mc=100):
    a_cd, a_sr = [], []
    for δ in deltas:
        ts = np.arange(t0+δ, T+1e-8, δ)
        errs_cd, errs_sr = [], []
        for _ in range(n_mc):
            x_true, zs = simulate_true_and_measure(ts, sigma)
            H = np.ones((2,2)) / (1 + sigma)
            R = np.eye(2) * sigma**2

            xh_cd = cd_ekf(zs, ts, H, R)
            xh_sr = sr_cd_ekf(zs, ts, H, R)

            errs_cd.append(np.sum((xh_cd - x_true)**2))
            errs_sr.append(np.sum((xh_sr - x_true)**2))

        K = len(ts)
        a_cd.append(np.sqrt(np.mean(errs_cd) / (2*K)))
        a_sr.append(np.sqrt(np.mean(errs_sr) / (2*K)))
    return np.array(a_cd), np.array(a_sr)

if __name__ == "__main__":
    sigma_list = [1e-4, 1e-6, 1e-8, 1e-10]
    delta_list = np.arange(0.1, 1.01, 0.1)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, σ in zip(axs.ravel(), sigma_list):
        arm_cd, arm_sr = compute_armse(σ, delta_list)
        ax.plot(delta_list, arm_cd, 'o-', label='CD-EKF')
        ax.plot(delta_list, arm_sr, 's-', label='SR-CD-EKF')
        ax.set_title(r'$\sigma={:.0e}$'.format(σ))
        ax.set_xlabel(r'$\delta$')
        ax.set_ylabel('ARMSE')
        ax.grid(True)
        ax.legend()
    plt.tight_layout()
    plt.show()
