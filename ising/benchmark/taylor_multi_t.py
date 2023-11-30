import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ising.hamiltonian import parametrized_ising
from ising.benchmark.sim_function import taylor_gate_count


def plot_gate_error(
    qubit,
    h_val,
    start_err_exp,
    end_err_exp,
    point_count,
    obs_norm,
):
    fig, ax = plt.subplots()

    configs = {
        "10": {"color": "blue", "label": "t=10"},
        "50": {"color": "black", "label": "t=50"},
        "100": {"color": "red", "label": "t=100"},
    }

    error_points = [
        10**x for x in np.linspace(start_err_exp, end_err_exp, point_count)
    ]
    ham = parametrized_ising(qubit, h_val)

    for time, config in configs.items():
        results = []

        for error in error_points:
            results.append(taylor_gate_count(ham, int(time), error, obs_norm))

        sns.lineplot(
            y=results,
            x=error_points,
            ax=ax,
            label=config["label"],
            color=config["color"],
        )

        sns.scatterplot(y=results, x=error_points, ax=ax, color=config["color"])

    ax.invert_xaxis()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\log_{10}(\epsilon)$")
    ax.set_ylabel(r"$\log_{10}(\text{gate count})$")

    ax.legend(loc="upper right", framealpha=1)
    ax.set_title("Depth vs Error for Different Time")

    diagram_name = "plots/benchmark/taylor_multi_t.png"
    print(f"Saving diagram at:{diagram_name}")
    plt.savefig(diagram_name, dpi=300)


if __name__ == "__main__":
    qubit = 25
    h_val = 0.1
    start_err_exp = -1
    end_err_exp = -3
    point_count = 10
    obs_norm = 1
    plot_gate_error(qubit, h_val, start_err_exp, end_err_exp, point_count, obs_norm)
