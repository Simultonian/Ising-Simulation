import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ising.hamiltonian import parametrized_ising_two


def plot_spectral_gap(h_val: float, start_side: int, end_side: int, point_count: int):
    fig, ax = plt.subplots()

    side_points = [int(x) for x in np.linspace(start_side, end_side, point_count)]

    gap = [
        parametrized_ising_two(side, h_val).approx_spectral_gap for side in side_points
    ]
    qubits = [side**2 for side in side_points]
    sns.lineplot(
        y=gap,
        x=qubits,
        ax=ax,
    )

    sns.scatterplot(y=gap, x=qubits, ax=ax)

    # ax.invert_xaxis()
    # ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\text{qubits }(N)$")
    ax.set_ylabel(r"Spectral gap")

    ax.set_title("Spectral vs Qubits for GSP on 2D Ising Model")

    ax.legend(loc="upper left", framealpha=1)
    # plt.legend()
    diagram_name = "plots/benchmark/spectral_gap_2d.png"
    print(f"Saving diagram at:{diagram_name}")
    plt.savefig(diagram_name, dpi=300)


def plot_spectral_across_h(side: int, start_h: float, end_h: float, point_count: int):
    fig, ax = plt.subplots()

    h_vals = [x for x in np.linspace(start_h, end_h, point_count)]

    gap = [parametrized_ising_two(side, h).approx_spectral_gap for h in h_vals]
    sns.lineplot(
        y=gap,
        x=h_vals,
        ax=ax,
    )

    sns.scatterplot(y=gap, x=h_vals, ax=ax)

    # ax.invert_xaxis()
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.set_xlabel(r"$\text{Transverse Field }(h)$")
    ax.set_ylabel(r"Spectral gap")

    ax.set_title("Spectral vs Transverse Field for GSP on 2D Ising Model")

    ax.legend(loc="upper left", framealpha=1)
    # plt.legend()
    diagram_name = "plots/benchmark/spectral_gap_2d.png"
    print(f"Saving diagram at:{diagram_name}")
    plt.savefig(diagram_name, dpi=300)


if __name__ == "__main__":
    start_h, end_h = -3, 3
    side = 2
    point_count = 30

    plot_spectral_across_h(side, start_h, end_h, point_count)
