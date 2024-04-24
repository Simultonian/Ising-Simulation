import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ising.hamiltonian import Hamiltonian, parse
import ising.benchmark.groundstate_depth as groundstate_depth

from typing import Callable

methods: dict[str, Callable[[Hamiltonian, float, float, float], int]] = {
    "Ge et al": groundstate_depth.ge_et_al,
    "Phase Estimation and Amplitude Amplification": groundstate_depth.phase_estimation,
    "Poulin et al": groundstate_depth.poulin_et_al,
    "Lin et al": groundstate_depth.lin_et_al,
    "Truncated Taylor Series": groundstate_depth.truncated_taylor,
    "First Order Trotter": groundstate_depth.first_order_trotter,
    "qDRIFT": groundstate_depth.qdrift,
}

colors: dict[str, str] = {
    "Ge et al": "blue",
    "Phase Estimation and Amplitude Amplification": "red",
    "Poulin et al": "black",
    "Lin et al": "red",
    "Truncated Taylor Series": "blue",
    "First Order Trotter": "purple",
    "qDRIFT": "yellow",
}


def plot_gate_error(
    name, molecule, start_err_exp, end_err_exp, point_count, obs_norm, eeta
):
    fig, ax = plt.subplots()

    configs = {
        "taylor": {"color": "blue", "label": "Truncated Taylor"},
        "qdrift": {"color": "red", "label": "qDRIFT Protocol"},
        "trotter": {"color": "black", "label": "First Order Trotter"},
    }

    error_points = [
        10**x for x in np.linspace(start_err_exp, end_err_exp, point_count)
    ]
    trotter, taylor, qdrift = [], [], []

    for eps in error_points:
        taylor.append(groundstate_depth.truncated_taylor(molecule, eeta, eps, obs_norm))
        qdrift.append(groundstate_depth.qdrift(molecule, eeta, eps, obs_norm))
        trotter.append(
            groundstate_depth.first_order_trotter(molecule, eeta, eps, obs_norm)
        )

    results = {"taylor": taylor, "qdrift": qdrift, "trotter": trotter}

    for method, config in configs.items():
        result = results[method]
        sns.lineplot(y=result, x=error_points, ax=ax, color=config["color"], alpha=0.5)
        sns.scatterplot(
            y=result,
            x=error_points,
            ax=ax,
            color=config["color"],
            label=configs[method]["label"],
        )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.invert_xaxis()

    ax.set_xlabel(r"Error ($\log_{10}$ Scale)")
    ax.set_ylabel(r"Gate Depth ($\log_{10}$ Scale)")

    # SETTING: AXIS VISIBILITY
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize=10)
    diagram_name = f"plots/benchmark/molecule/eps_gate/{name}_trotter_qdrift_tts.png"
    print(f"Saving diagram at:{diagram_name}")
    plt.savefig(diagram_name, dpi=300)


if __name__ == "__main__":
    name = "methane"
    molecule = parse(name)
    start_err_exp = -1
    end_err_exp = -5
    eeta = 0.8
    point_count = 10
    obs_norm = 1

    plot_gate_error(
        name, molecule, start_err_exp, end_err_exp, point_count, obs_norm, eeta
    )
