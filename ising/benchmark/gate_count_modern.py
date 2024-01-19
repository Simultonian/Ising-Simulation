import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ising.hamiltonian import parametrized_ising, Hamiltonian
import ising.benchmark.groundstate_depth as groundstate_depth

from typing import Callable

methods: dict[str, Callable[[Hamiltonian, float, float, float], int]] = {
    "Ge et al": groundstate_depth.ge_et_al,
    # "Phase Estimation and Amplitude Amplification": groundstate_depth.phase_estimation,
    # "Poulin et al": groundstate_depth.poulin_et_al,
    "Lin et al": groundstate_depth.lin_et_al,
    "Truncated Taylor Series": groundstate_depth.truncated_taylor,
    # "First Order Trotter": groundstate_depth.first_order_trotter,
    # "qDRIFT": groundstate_depth.qdrift,
}

colors: dict[str, str] = {
    "Ge et al": "blue",
    "Phase Estimation and Amplitude Amplification": "red",
    "Poulin et al": "black",
    "Lin et al": "green",
    "Truncated Taylor Series": "black",
    "First Order Trotter": "purple",
    "qDRIFT": "yellow",
}


def plot_qubit_gate_count(
    h_val: float,
    start_qubit: int,
    end_qubit: int,
    obs_norm: float,
    eps: float,
    eeta: float,
    point_count: int,
):
    fig, ax = plt.subplots()

    qubit_points = [int(x) for x in np.linspace(start_qubit, end_qubit, point_count)]

    results: dict[str, list[int]] = {}
    for method in methods.keys():
        results[method] = []

    for qubit in qubit_points:
        ham = parametrized_ising(qubit, h_val)
        for method, fn in methods.items():
            result = fn(ham, eeta, eps, obs_norm)
            results[method].append(result)

    for method in methods.keys():
        result = results[method]
        sns.lineplot(
            y=result,
            x=qubit_points,
            ax=ax,
            label=method,
            color=colors[method],
            alpha=0.6,
        )

        sns.scatterplot(y=result, x=qubit_points, ax=ax, color=colors[method])

    # ax.invert_xaxis()
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.set_xlabel(r"$\text{qubits }(N)$")
    ax.set_ylabel(r"$\log_{10}(\text{gate count})$")

    ax.set_title("Gate Count vs Qubits for Simulation Techniques")

    ax.legend(loc="upper left", framealpha=1)
    # plt.legend()
    diagram_name = "plots/benchmark/gate_modern.png"
    print(f"Saving diagram at:{diagram_name}")
    plt.savefig(diagram_name, dpi=300)


if __name__ == "__main__":
    h_val = 1
    start_qubit, end_qubit = 4, 10
    point_count = 10
    obs_norm = 1
    eps = 1e-3
    eeta = 0.6

    plot_qubit_gate_count(
        h_val, start_qubit, end_qubit, obs_norm, eps, eeta, point_count
    )
