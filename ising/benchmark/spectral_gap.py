import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ising.hamiltonian import parametrized_ising


def plot_spectral_gap(h_val: float, start_qubit: int, end_qubit: int, point_count: int):
    fig, ax = plt.subplots()

    qubit_points = [int(x) for x in np.linspace(start_qubit, end_qubit, point_count)]

    gap = [parametrized_ising(qubit, h_val).spectral_gap for qubit in qubit_points]
    sns.lineplot(
        y=gap,
        x=qubit_points,
        ax=ax,
    )

    sns.scatterplot(y=gap, x=qubit_points, ax=ax)

    # ax.invert_xaxis()
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.set_xlabel(r"$\text{qubits }(N)$")
    ax.set_ylabel(r"Spectral gap")

    ax.set_title("Spectral vs Qubits for Simulation Techniques")

    ax.legend(loc="upper left", framealpha=1)
    # plt.legend()
    diagram_name = "plots/benchmark/spectral_gap.png"
    print(f"Saving diagram at:{diagram_name}")
    plt.savefig(diagram_name, dpi=300)


if __name__ == "__main__":
    h_val = 1
    start_qubit, end_qubit = 4, 10
    point_count = 10

    plot_spectral_gap(h_val, start_qubit, end_qubit, point_count)
