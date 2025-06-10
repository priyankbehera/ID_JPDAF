def logpdf(x, mu, B, v):
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
    v : (k,) array
        The conditional variances.

    Returns
    -------
    float
        log p(x | mu, B, v).
    """
    diff = x - mu
    I = np.eye(B.shape[0])
    # precision matrix
    Lambda = (I - B).T @ np.diag(1.0 / v) @ (I - B)

    # log determinant of Sigma_ID is sum log v_i
    log_det = np.sum(np.log(v))

    k = x.shape[0]
    quad = diff.T @ Lambda @ diff

    return -0.5 * (log_det + quad + k * np.log(2 * np.pi))