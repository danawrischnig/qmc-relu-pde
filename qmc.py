import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

def qmc_point(i, z, n, shift):
    """Randomly shifted rank-1 lattice point in [0, 1)^s."""
    return np.mod(i * z / n + shift, 1.0)

def qmc_values(integrand, z, n, shift, parallel):
    """Return array of QoI evaluations for one randomly shifted lattice."""

    values = parallel(
        delayed(integrand)(qmc_point(i, z, n, shift))
        for i in range(n)
    )

    return np.asarray(values, dtype=float)

# Off the shelf generating vector for rank-1 lattice rules
generator = np.loadtxt("https://vkaarnioja.github.io/offtheshelf2048.txt") 

def qmc_estimates(
    integrand,
    s,
    log_n,
    log_R=4,
    rng=None,
    n_jobs=-1,
    show_progress=True,
    desc = "Random shifted rank-1-lattice rule",
):
    """
    Run a randomized QMC experiment for a scalar- or vector-valued integrand.

    For each sample size, the returned estimate is the mean over independent
    random shifts. The randomized-QMC standard error is estimated from the
    variation between these randomizations.

    Parameters
    ----------
    integrand : callable
        Function mapping y in [0, 1]^dim to a scalar or vector.

    dim : int
        Dimension of the integration domain.

    log_n : int
        Largest sample size is 2^log_n.

    log_R : int, default=4
        Number of randomizations is 2^log_R.

    rng : np.random.Generator, optional
        Random number generator. If None, a new generator is created.

    n_jobs : int, default=-1
        Number of parallel jobs used for integrand evaluations.

    Returns
    -------
    sample_sizes : ndarray
        Sample sizes
            2^2, 2^3, ..., 2^log_sample_size.

    estimates : ndarray
        Mean randomized-QMC estimates with shape
            (output_dim, len(sample_sizes)).

    standard_errors : ndarray
        Estimated standard errors of the mean over random shifts, with shape
            (output_dim, len(sample_sizes)).
    """
    s = int(s)
    log_n = int(log_n)
    log_R = int(log_R)
    n_jobs = int(n_jobs)

    # ------------------------------------------------
    # Validate input parameters
    # ------------------------------------------------
    if integrand is None or not callable(integrand):
        raise ValueError("integrand must be a callable function.")

    if s < 1:
        raise ValueError(f"dimension s must be a positive integer, but got {s}.")
    
    if log_n < 4:
        raise ValueError(f"log_n must be at least 4 to ensure a minimum sample size of 16, but got {log_n}.")
    
    if log_R < 1:
        raise ValueError(f"log_R must be at least 1 to ensure a minimum of 2 random shifts, but got {log_R}.")

    if rng is not None and not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy.random.Generator instance.")

    if n_jobs < -1 or n_jobs == 0:
        raise ValueError(f"n_jobs must be a positive integer or -1, but got {n_jobs}.")

    #------------------------------------------------
    # Prepare for the QMC experiment
    #------------------------------------------------
    z = generator[:s]

    sample_sizes = 2 ** np.arange(
        2,
        log_n + 1,
        dtype=int,
    )
    n_max = sample_sizes[-1]

    R = 2 ** log_R  # number of random shifts

    if rng is None:
        rng = np.random.default_rng()

    shifted_estimates = []

    with Parallel(n_jobs=n_jobs) as parallel:
        for _ in tqdm(
            range(R),
            desc=desc,
            unit="shifts",
            disable=not show_progress,
        ):
            shift = rng.uniform(0.0, 1.0, s)

            values_max = qmc_values(
                integrand,
                z,
                n_max,
                shift,
                parallel,
            )

            values_max = np.asarray(values_max, dtype=float)

            if values_max.ndim == 1:
                values_max = values_max[:, None]

            estimates_r = []

            for n in sample_sizes:
                stride = n_max // n
                values = values_max[::stride]

                estimates_r.append(
                    np.mean(values, axis=0)
                )

            shifted_estimates.append(
                np.stack(estimates_r, axis=1)
            )

    # Shape: (randomizations, output_dim, number of sample sizes)
    shifted_estimates = np.stack(
        shifted_estimates,
        axis=0,
    )

    # Mean over independent random shifts
    averages = np.mean(
        shifted_estimates,
        axis=0,
    )

    # Standard error of that mean
    standard_errors = (
        np.std(
            shifted_estimates,
            axis=0,
            ddof=1,
        )
        / np.sqrt(R)
    )

    return sample_sizes, averages, standard_errors

def estimate_exceedance_threshold(
    g,
    s,
    log_n,
    exceedance,
    lower_exceedance=None,
    log_R=4,
    rng=None,
    n_jobs=-1,
):
    """
    Estimate a threshold whose empirical exceedance probability is at most
    ``exceedance``.

    The function generates ``randomizations`` randomized QMC samples, each
    containing ``sample_size`` function evaluations, and pools the resulting
    values. It then selects an empirical quantile ``b`` such that

        P(values > b) <= exceedance.

    If ``lower`` is provided, the empirical exceedance probability is also
    checked against the lower bound. A warning is printed if

        P(values > b) < lower.

    Parameters
    ----------
    g : array-like
        Generator or generating vector used by the QMC rule.
    s : int
        Number of dimensions to use.
    n : int
        Number of QMC samples generated per randomization.
    exceedance : float, default=0.01
            Maximum allowed empirical exceedance probability. Must lie in
            between 0 and 1.
    lower_exceedance : float, optional
        Optional lower bound on the empirical exceedance probability.
        Must satisfy ``0 <= lower <= upper``.
    randomizations : int, default=2**6
        Number of independent random shifts used to randomize the QMC rule.
    random_seed : int or None, optional
        Seed for the random number generator used to construct the random
        shifts.
    n_jobs : int, default=-1
        Number of parallel jobs used during function evaluation. The value
        ``-1`` typically means using all available processors.

    Returns
    -------
    b : float
        Estimated threshold satisfying the empirical upper exceedance
        constraint.

    Notes
    -----
    The threshold is computed using NumPy's ``"inverted_cdf"`` quantile
    method. Consequently, ``b`` is one of the observed sample values rather
    than an interpolated value.

    Because the empirical distribution is discrete, especially in the
    presence of ties, it may be impossible to obtain an exceedance
    probability within the interval ``[lower_exceedance, exceedance]``. In that case, the
    upper constraint is retained and the lower constraint may be violated.
    """
    s = int(s)
    log_n = int(log_n)
    log_R = int(log_R)
    n_jobs = int(n_jobs)

    # ------------------------------------------------
    # Validate input parameters
    # ------------------------------------------------
    if g is None or not callable(g):
        raise ValueError("integrand must be a callable function.")

    if s < 1:
        raise ValueError(f"dimension s must be a positive integer, but got {s}.")
    
    if log_n < 4:
        raise ValueError(f"log_n must be at least 4 to ensure a minimum sample size of 16, but got {log_n}.")
    
    if log_R < 1:
        raise ValueError(f"log_R must be at least 1 to ensure a minimum of 2 random shifts, but got {log_R}.")

    if rng is not None and not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy.random.Generator instance.")

    if n_jobs < -1 or n_jobs == 0:
        raise ValueError(f"n_jobs must be a positive integer or -1, but got {n_jobs}.")

    #------------------------------------------------
    # Prepare for the QMC experiment
    #------------------------------------------------
    z = generator[:s]

    n = 2 ** log_n  # number of QMC samples per randomization
    R = 2 ** log_R  # number of random shifts

    if rng is None:
        rng = np.random.default_rng()

    
    rng = np.random.default_rng() if rng is None else rng

    values = []

    with Parallel(n_jobs=n_jobs,) as parallel:
        for r in tqdm(
            range(R),
            desc="Threshold randomizations",
        ):
            shift = rng.uniform(0.0, 1.0, s)

            vals = qmc_values(
                g,
                z,
                n,
                shift,
                parallel,
            )

            values.extend(vals)

    threshold = np.quantile(
        values,
        1 - exceedance,
        method="inverted_cdf",
    )

    exceedance = np.mean(values > threshold)

    if lower_exceedance is not None and exceedance < lower_exceedance:
        print(
            f"Warning: The estimated quantile {threshold:.4e} "
            f"has an empirical exceedance probability "
            f"{exceedance:.4%}, which is below the specified "
            f"lower bound {lower_exceedance:.4%}."
        )

    return threshold