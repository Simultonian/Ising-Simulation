import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ising.hamiltonian import parametrized_ising, Hamiltonian
import ising.benchmark.groundstate_depth as groundstate_depth

from typing import Callable

methods: dict[str, Callable[[Hamiltonian, float, float, float], int]] = {
    # "Ge et al": groundstate_depth.ge_et_al,
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
    "Lin et al": "red",
    "Truncated Taylor Series": "blue",
    "First Order Trotter": "purple",
    "qDRIFT": "yellow",
}


def plot_qubit_gate_count(
    h_vals: list[list[float]],
    start_qubit: int,
    end_qubit: int,
    obs_norm: float,
    eps: float,
    eeta: float,
    point_count: int,
):
    row, col = len(h_vals), len(h_vals[0])
    fig, axes = plt.subplots(row, col, sharex=True)

    for r, h_row in enumerate(h_vals):
        for c, h_val in enumerate(h_row):
            print(f"Running for h={h_val}")

            ax = axes[r, c]
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

                sns.scatterplot(y=result, x=qubit_points, ax=ax, color=colors[method], s=5)

            # ax.invert_xaxis()
            # ax.set_xscale("log")
            ax.set_yscale("log")
            # ax.set_ylabel(r"$\log_{10}(\text{gate count})$")
            ax.get_legend().remove()
            ax.set_title(f"h={h_val}")

    # fig.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    # ax.legend(loc='upper right')
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes][:1]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels)

    fig.tight_layout()
    # fig.xlabel(r"$\text{qubits}(N)$")
    # fig.ylabel("Gate Count")
    # plt.set_title("Gate Count vs Qubits for Simulation Techniques")
    # ax.legend(loc="upper left", framealpha=1)
    # plt.legend()
    fig.supxlabel(r"$\text{qubits}(N)$")
    fig.supylabel("Gate Count")
    diagram_name = "plots/benchmark/multi_modern.png"
    print(f"Saving diagram at:{diagram_name}")
    plt.savefig(diagram_name, dpi=300)


def main():
    # h_vals = [[0.01, 0.1, 0.9], [1.01, 1.1, 10]]
    h_vals = [[0.9, 0.99, 1.01], [0.1, 0.8, 2]]
    start_qubit, end_qubit = 20, 200
    point_count = 10
    obs_norm = 1
    eps = 1e-3
    eeta = 0.2

    plot_qubit_gate_count(
        h_vals, start_qubit, end_qubit, obs_norm, eps, eeta, point_count
    )


def test_main():
    main()

if __name__ == "__main__":
    main()
