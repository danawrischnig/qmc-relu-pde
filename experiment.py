import numpy as np
import json
from qmc import qmc_estimates, estimate_exceedance_threshold
import argparse
from tqdm import tqdm
from plot import plot_results


def sp(x, theta):
    """
    Scaled softplus function: log(1 + exp(theta * x)) / theta.

    Parameters
    ----------
    x : array-like
        Input values.
    theta : float
        Scaling parameter for the softplus function.    

    Returns
    -------
    softplus_values : array-like
        The computed softplus values for the input x.
    """
    return np.logaddexp(0.0, theta * np.asarray(x)) / theta

def relu(x):
    """
    ReLU function: max(0, x).

    Parameters
    ----------
    x : array-like
        Input values.

    Returns
    -------
    relu_values : array-like
        The computed ReLU values for the input x.
    """
    return np.maximum(0.0, x)


def get_g(pde):
    """
    Returns a function g(y) that solves the PDE for a given parameter y and integrates the solution.

    Parameters
    ----------
    pde : object
        An instance of a PDE class that has a `solve` method and an `integrate` method.

    Returns
    -------
    g : function
        A function that takes a parameter y, solves the PDE, and returns the integrated solution.
    """
    def g(y):
        u = pde.solve(y)
        return u.integrate()
    return g

def get_integrad(pde, theta_values):
    """
    Returns a function that computes the integrand for the QMC experiment, which includes the ReLU and Softplus approximations of the PDE solution.
    
    Parameters
    ----------
    pde : object
        An instance of a PDE class that has a `solve` method and an `integrate` method.
    theta_values : array-like
        An array of theta values for the Softplus function.
    """
    g = get_g(pde)

    outer_functionals = lambda x: np.array([relu(x)] +[sp(x, theta=theta) for theta in theta_values])

    def integrand(y):
        return outer_functionals(g(y))
    
    return integrand


def round_sig(x, digits=2):
    """
    Round a number to a specified number of significant digits.

    Parameters
    ----------
    x : float
        The number to be rounded.
    digits : int
        The number of significant digits.

    Returns
    -------
    rounded_x : float
        The rounded number.
    """
    if x == 0:
        return 0.0
    else:
        return round(x, digits - int(np.floor(np.log10(abs(x)))) - 1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the QMC experiment for the parametric PDE."
    )

    # PDE parameters
    parser.add_argument(
        "--pde-name",
        type=str,
        default="constant_diffusion",
        choices=[
            "constant_diffusion",
            "sine_diffusion",
            "parabolic_sine_diffusion",
        ],
        help="PDE/diffusion coefficient type.",
    )
    parser.add_argument(
        "--baseline",
        type=float,
        default=None,
        help="Baseline value for the diffusion coefficient.",
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=1.5,
        help="Decay parameter for the diffusion coefficient.",
    )
    parser.add_argument(
        "--s",
        type=int,
        default=20,
        help="Dimension/truncation parameter.",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=5,
        help="FEM discretization level; each spatial axis is divided into 2**level parts.",
    )

    # Quantities of interest parameters
    parser.add_argument(
        "--log-theta",
        type=int,
        default=12,
        help="Theta ranges from 1 to 2**log_theta.",
    )
    

    # QMC experiment parameters
    parser.add_argument(
            "--log-n",
            type=int,
            default=13,
            help="Largest sample size: n = 2**log_n.",
        )
    parser.add_argument(
            "--log-n-ref",
            type=int,
            default=18,
            help="Reference sample size for estimating the exceedance threshold: n_ref = 2**log_n_ref.",
        )

    parser.add_argument(
        "--log-R",
        type=int,
        default=8,
        help="Number of randomizations: R = 2**log_R.",
    )
    parser.add_argument(
        "--log-R-ref",
        type=int,
        default=8,
        help="Reference number of randomizations for estimating the exceedance threshold: R_ref = 2**log_R_ref.",
    )

    parser.add_argument(
        "--log-n-batches",
        type=int,
        default=5,
        help="Number of batches: number of batches = 2**log_n_batches.",
    )
    
    # General parameters
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="Number of parallel jobs.",
    )

    # File names
    parser.add_argument(
        "--filename-data",
        type=str,
        default=None,
        help="Name of the data file to save.",
    )

    parser.add_argument(
        "--filename-params",
        type=str,
        default=None,
        help="Name of the parameters file to save.",
    )
    parser.add_argument(
        "--filename-plot",
        type=str,
        default=None,
        help="Name of the plot file to save.",
    )

    args = parser.parse_args()
    validate_args(args, parser)

    return args

def validate_args(args, parser):
    if args.baseline is not None and args.baseline <= 0:
        parser.error("--baseline must be positive.")

    if args.decay <= 1:
        parser.error("--decay must be greater than 1.")

    if args.s < 1:
        parser.error("--s must be at least 1.")

    if args.level < 2:
        parser.error("--level must be at least 2.")

    if args.log_theta < 0:
        parser.error("--log-theta must be non-negative.")

    if args.log_R < 1:
        parser.error("--log-R must be at least 1.")

    if args.log_R_ref < args.log_R:
        parser.error("--log-R-ref must be greater than or equal to --log-R.")

    if args.log_n < 5:
        parser.error("--log-n must be at least 5.")

    if args.log_n_ref <= args.log_n:
        parser.error("--log-n-ref must be greater than --log-n.")

    if args.log_n_batches < 1:
        parser.error("--log-n-batches must be at least 1.")

    if args.n_jobs != -1 and args.n_jobs < 1:
        parser.error("--n-jobs must be -1 or a positive integer.")


def main():
    args = parse_args()

    pde_name = args.pde_name
    decay = args.decay
    baseline = args.baseline 
    if baseline is None:
        if pde_name == "constant_diffusion":
            baseline = 1.0
        elif pde_name == "sine_diffusion":
            baseline = round_sig(np.sum(np.arange(1, args.s + 1) ** (-decay)))
        elif pde_name == "parabolic_sine_diffusion":
            raise NotImplementedError("Parabolic sine diffusion PDE is not (correctly)implemented yet.")
        else:
            raise ValueError(f"Unknown PDE name: {pde_name}. Please specify a baseline value using --baseline.")
    

    s = args.s
    level = args.level
    
    log_theta = args.log_theta

    log_R_ref = args.log_R_ref
    log_R = args.log_R
    log_n = args.log_n
    log_n_ref = args.log_n_ref
    log_n_batches = args.log_n_batches

    random_seed = args.random_seed
    n_jobs = args.n_jobs

    filename_data = args.filename_data or (
        f"data_{pde_name}_decay_{decay}_baseline_{baseline}_"
        f"s{s}_log_R{log_R}_log_n_ref{log_n_ref}_log_n_batches{log_n_batches}.npz"
    )
    filename_params = args.filename_params or (
        f"params_{pde_name}_decay_{decay}_baseline_{baseline}_"
        f"s{s}_log_R{log_R}_log_n_ref{log_n_ref}_log_n_batches{log_n_batches}.json"
    )
    filename_plot = args.filename_plot or (
        f"plot_{pde_name}_decay_{decay}_baseline_{baseline}_"
        f"s{s}_log_R{log_R}_log_n_ref{log_n_ref}_log_n_batches{log_n_batches}.png"
    )
    
    # Set up the PDE problem based on the specified name
    if pde_name == "constant_diffusion":
        from parametric_pde import ConstantDiffusion
        pde = ConstantDiffusion(dim=s, decay=decay, baseline=baseline, boundary_data=0.0)
    elif pde_name == "sine_diffusion":
        from parametric_pde import SineDiffusion
        pde = SineDiffusion(dim=s, level=level, baseline=baseline, decay=decay, boundary_data=0.0)
    elif pde_name == "parabolic_sine_diffusion":
        raise NotImplementedError("Parabolic sine diffusion PDE is not (correctly)implemented yet.")
    else:
        # Placeholder for another PDE implementation
        raise NotImplementedError("Another PDE is not implemented yet.")

    # Integrands
    g = get_g(pde)

    theta_values = 2 ** np.arange(log_theta + 1)
    integrand = get_integrad(pde, theta_values)

    # Random number generator
    rng = np.random.default_rng(random_seed)

    # Compute boundary data for ReLU integrands peforming badly with qmc
    # Choose boundary data such that only ~1% of the QMC values are positive
    threshold = estimate_exceedance_threshold(
        g=g,
        s=s,
        log_n=8,
        log_R=4,
        exceedance=0.01,
        rng=rng,
        n_jobs=n_jobs,
    )
    boundary_data = - round_sig(threshold)
    pde.set_boundary_data(boundary_data)

    experiment_params = {
        "pde_name": pde_name,
        "baseline": baseline,
        "decay": decay,
        "boundary_data": boundary_data,
        "s": s,
        "log_theta": log_theta,
        "log_n": log_n,
        "log_R": log_R,
        "log_n_ref": log_n_ref,
        "log_R_ref": log_R_ref,
        "log_n_batches": log_n_batches,
        "random_seed": random_seed,
        "n_jobs": n_jobs,
    }

    print("Experiment parameters:")
    for key, value in experiment_params.items():
        print(f"{key}: {value}")

    # Compute reference values
    _, references, _ = qmc_estimates(
        integrand=integrand,
        s=s,
        log_n=log_n_ref,
        log_R=log_R_ref,
        rng=rng,
        n_jobs=n_jobs,
        desc="Computing qmc estimates for reference values",
    )
    references = references[:, -1]

    # Run randomized QMC experiment
    estimates = []
    standard_errors = []

    for _ in tqdm(range(2**log_n_batches), desc="Computing qmc estimates for each batch/trial", unit="batch"):
        sample_sizes, batch_estimates, batch_standard_errors = qmc_estimates(
            integrand=integrand,
            s=s,
            log_n=log_n,
            log_R=log_R,
            rng=rng,
            n_jobs=n_jobs,
            desc="Computing qmc estimates for each batch/trial",
            show_progress=False,
        )
        estimates.append(batch_estimates)
        standard_errors.append(batch_standard_errors)

    estimates = np.array(estimates, dtype=float) # shape: (number of batches, output_dim, number of sample sizes)
    standard_errors = np.array(standard_errors, dtype=float) # shape: (number of batches, output_dim, number of sample sizes)
    
    experiment_data = {
        "sample_sizes": sample_sizes,
        "theta_values": theta_values,
        "relu_estimates": estimates[:, 0],
        "relu_reference": references[0],
        "softplus_estimates": estimates[:, 1:],
        "softplus_references": references[1:],
        "relu_standard_errors": standard_errors[:, 0],
        "softplus_standard_errors": standard_errors[:, 1:],
    }

    fig = plot_results(experiment_data, experiment_params)

    # Save experiment parameters to a JSON file
    with open(filename_params, "w") as f:
        json.dump(experiment_params, f)   
    print(f"Experiment parameters saved to {filename_params}")

    # Save experiment data to a .npz file
    np.savez(filename_data, **experiment_data)
    print(f"Experiment data saved to {filename_data}")

    # Save the figure to a file
    fig = plot_results(experiment_data, experiment_params)

    fig.savefig(
        filename_plot,
        bbox_inches="tight",
    )

if __name__ == "__main__":
    main()
