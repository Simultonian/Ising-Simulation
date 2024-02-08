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
    }

    error_points = [
        10**x for x in np.linspace(start_err_exp, end_err_exp, point_count)
    ]
    taylor, qdrift = [], []

    for eps in error_points:
        taylor.append(groundstate_depth.truncated_taylor(molecule, eeta, eps, obs_norm))
        qdrift.append(groundstate_depth.qdrift(molecule, eeta, eps, obs_norm))

    results = {"taylor": taylor, "qdrift": qdrift}

    for method, config in configs.items():
        result = results[method]
        sns.lineplot(
            y=result,
            x=error_points,
            ax=ax,
            label=configs[method]["label"],
            color=config["color"],
        )
        sns.scatterplot(y=result, x=error_points, ax=ax, color=config["color"])

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.invert_xaxis()

    ax.set_xlabel(r"$\log_{10}(\epsilon)$")
    ax.set_ylabel(r"$\log_{10}(\text{gate count})$")

    ax.set_title("Groundstate Preparation of Methane", pad=20)

    plt.legend()
    diagram_name = f"plots/benchmark/molecule/{name}_qdrift_tts.png"
    print(f"Saving diagram at:{diagram_name}")
    plt.savefig(diagram_name, dpi=300)


if __name__ == "__main__":
    name = "methane"
    molecule = parse(name)
    start_err_exp = -1
    end_err_exp = -3
    eeta = 0.8
    point_count = 10
    obs_norm = 1

    plot_gate_error(
        name, molecule, start_err_exp, end_err_exp, point_count, obs_norm, eeta
    )
