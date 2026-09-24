import numpy as np

def powerfit(n, error):
    """Fit error approximately C*n**r by least squares in log-log coordinates.
    
    Supports both scalar and vector-valued errors.
    
    Parameters
    ----------
    n : array_like
        Sample sizes.
    error : array_like
        RMS errors. Shape (len(n),) for scalar or (N, len(n)) for vector.
    
    Returns
    -------
    C : float or ndarray
        Constant. Scalar if error is 1D, shape (N,) if error is 2D.
    r : float or ndarray
        Power law exponent. Scalar if error is 1D, shape (N,) if error is 2D.
    """
    n = np.asarray(n, dtype=float)
    error = np.asarray(error, dtype=float)
    
    log_n = np.log(n)
    log_error = np.log(error)
    
    if error.ndim == 1:
        # Scalar case
        r, log_C = np.polyfit(log_n, log_error, 1)
        return np.exp(log_C), r
    else:
        # Vector case: error shape (N, len(sample_sizes))
        N = error.shape[0]
        C = np.empty(N)
        r = np.empty(N)
        for i in range(N):
            r[i], log_C = np.polyfit(log_n, log_error[i], 1)
            C[i] = np.exp(log_C)
        return C, r
