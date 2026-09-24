import numpy as np
from powerfit import powerfit
from matplotlib import pyplot as plt

def plot_results(experiment_data, experiment_params):
    """Plot QMC and total RMSE for ReLU and Softplus."""

    sample_sizes = np.asarray(experiment_data["sample_sizes"])
    theta_values = np.asarray(experiment_data["theta_values"])

    relu_estimates = np.asarray(experiment_data["relu_estimates"])
    relu_reference = experiment_data["relu_reference"]

    softplus_estimates = np.asarray(experiment_data["softplus_estimates"])
    softplus_references = np.asarray(experiment_data["softplus_references"])

    plot_indices = range(0, len(theta_values), 3)
    fit_slice = slice(-4, None)

    # Errors
    relu_rmse = np.sqrt(
        np.mean((relu_estimates - relu_reference) ** 2, axis=0)
    )
    softplus_qmc_rmse = np.sqrt(
        np.mean(
            (softplus_estimates - softplus_references[:, None]) ** 2,
            axis=0,
        )
    )
    softplus_total_rmse = np.sqrt(
        np.mean((softplus_estimates - relu_reference) ** 2, axis=0)
    )
    biases = np.abs(softplus_references - relu_reference)

    # Convergence rates
    constants_softplus, rates_softplus = powerfit(
        sample_sizes[fit_slice], softplus_qmc_rmse[:, fit_slice]
    )
    constant_relu, rate_relu = powerfit(
        sample_sizes[fit_slice], relu_rmse[fit_slice]
    )

    # Figure
    fig, (ax_qmc, ax_total) = plt.subplots(
        1, 2, figsize=(13, 5), sharex=True
    )

    # QMC error
    for i in plot_indices:
        theta = theta_values[i]
        
        C = constants_softplus[i]
        exponent = int(np.floor(np.log10(abs(C))))
        mantissa = C / 10**exponent

        label = (
            fr"$\Psi_\theta$, $\theta={theta:g}$, "
            fr"fit: ${mantissa:.1f}\times 10^{{{exponent}}}"
            fr"n^{{{rates_softplus[i]:.2f}}}$"
        )

        line, = ax_qmc.loglog(
            sample_sizes,
            softplus_qmc_rmse[i],
            "o",
            markersize=4,
            label=label,
        )

        ax_qmc.loglog(
            sample_sizes,
            constants_softplus[i] * sample_sizes**rates_softplus[i],
            "--",
            color=line.get_color(),
            linewidth=1.3,
        )
    exponent = int(np.floor(np.log10(abs(constant_relu))))
    mantissa = constant_relu / 10**exponent
    label_relu = (
        fr"ReLU fit: ${mantissa:.1f}\times 10^{{{exponent}}}"
        fr"n^{{{rate_relu:.2f}}}$"
    )
    ax_qmc.loglog(
        sample_sizes,
        relu_rmse,
        "o",
        color="gray",
        markersize=5,
        label=label_relu,
    )
    ax_qmc.loglog(
        sample_sizes,
        constant_relu * sample_sizes**rate_relu,
        color="gray",
        linewidth=1.3,
    )

    # Total error
    for i in plot_indices:
        theta = theta_values[i]

        line, = ax_total.loglog(
            sample_sizes,
            softplus_total_rmse[i],
            "o--",
            markersize=4,
            label=fr"$\Psi_\theta$, $\theta={theta:g}$ "
        )

        ax_total.axhline(
            biases[i],
            linestyle=":",
            color=line.get_color(),
            linewidth=1.3,
        )

    ax_total.loglog(
        sample_sizes,
        relu_rmse,
        "o-",
        color="gray",
        markersize=5,
        label="ReLU",
    )

    # Labels
    ax_qmc.set_title("QMC error")
    ax_qmc.set_ylabel("RMSE w.r.t. matching reference")

    ax_total.set_title("Total error")
    ax_total.set_ylabel("RMSE w.r.t. ReLU reference")

    for ax in (ax_qmc, ax_total):
        ax.set_xlabel(r"Number of QMC points $n$")
        ax.grid(True, which="both", linestyle=":", alpha=0.4)
        ax.legend(fontsize=8)

    # Experiment description
    name = experiment_params["pde_name"].replace("_", " ").title()

    params = (
        fr"$s={experiment_params['s']}$, "
        fr"$\alpha={experiment_params['decay']:g}$, "
        fr"$a_0={experiment_params['baseline']:g}$, "
        fr"$b={experiment_params['boundary_data']:g}$"
    )

    if experiment_params["pde_name"] == "sine_diffusion":
        if "level" in experiment_params:
            params += fr", FEM level $L={experiment_params['level']}$"

    fig.suptitle(
        f"{name}: ReLU vs. Softplus\n{params}",
        fontsize=13,
    )

    fig.tight_layout()
    return fig