import numpy as np

def logpdf(x, B, V):
    """
    Multi‐variate normal log‐pdf in the (B, v) influence‐diagram form.
    Parameters
    ----------
    x : (k,) array
        The point at which to evaluate.
    mu : (k,) array
        The mean vector.
    B : (k,k) array
        The ID arc‐coefficient matrix.
    V : (k,) array
        The conditional variances.

    Returns
    -------
    float
        log p(x | B, V).
    """
    print(x)
    print(B)
    print(V)
    print(np.diag(1.0 / V))
    print(np.diagflat(1.0 / V))

    I = np.eye(B.shape[0])
    # precision matrix
    Lambda = (I - B).T @ np.diagflat(1.0 / V) @ (I - B)

    # log determinant of Sigma_ID is sum log v_i
    log_det = np.sum(np.log(V))

    k = x.shape[0]
    quad = x.T @ Lambda @ x

    return -0.5 * (log_det + quad + k * np.log(2 * np.pi))