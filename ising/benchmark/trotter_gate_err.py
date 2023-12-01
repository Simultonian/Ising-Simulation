import matplotlib.pyplot as plt
import seaborn as sns

from ising.hamiltonian import parametrized_ising
from ising.benchmark.sim_function import (
    trotter_gate_count,
)


def plot_gate_error(paras):
    fig, ax = plt.subplots()

    qubit = paras["qubit"]
    h_val = paras["h_val"]
    ham = parametrized_ising(qubit, h_val)

    time = paras["time"]
    r_range = paras["r_range"]
    rel_diff = paras["rel_diff"]

    trotter = []
    for error in rel_diff:
        trotter.append(trotter_gate_count(ham, time, error))

    sns.lineplot(
        y=r_range,
        x=rel_diff,
        ax=ax,
    )
    sns.scatterplot(y=r_range, x=rel_diff, ax=ax, label="Numerical")

    sns.lineplot(
        y=trotter,
        x=rel_diff,
        ax=ax,
    )
    sns.scatterplot(y=trotter, x=rel_diff, ax=ax, label="Analytical")

    ax.invert_xaxis()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\log_{10}(\epsilon)$")
    ax.set_ylabel(r"$\log_{10}(\text{r})$")

    ax.legend(loc="upper left", framealpha=1)
    ax.set_title("First Order Trotter: Error Bound vs Numerical Results")

    diagram_name = "plots/benchmark/trotter_gate_count.png"
    print(f"Saving diagram at:{diagram_name}")
    plt.savefig(diagram_name, dpi=300)


from ising.utils import read_json

if __name__ == "__main__":
    qubit = 8
    file_name = f"data/simulation/benchmark_magnetization_lie_{qubit}.json"
    input_paras = read_json(file_name)

    plot_gate_error(input_paras)
