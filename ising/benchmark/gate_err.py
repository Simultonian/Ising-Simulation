import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ising.hamiltonian import parametrized_ising
from ising.benchmark.sim_function import (
    taylor_gate_count,
    trotter_gate_count,
    qdrift_gate_count,
)


def plot_gate_error(
    qubit, h_val, start_err_exp, end_err_exp, point_count, obs_norm, time
):
    fig, ax = plt.subplots()

    configs = {
        "taylor": {"color": "blue", "label": "Truncated Taylor"},
        "trotter": {"color": "black", "label": "First Order Trotter"},
        "qdrift": {"color": "red", "label": "qDRIFT Protocol"},
    }

    error_points = [
        10**x for x in np.linspace(start_err_exp, end_err_exp, point_count)
    ]
    taylor, trotter, qdrift = [], [], []
    ham = parametrized_ising(qubit, h_val)

    inv_err = [1 / x for x in error_points]

    for error in error_points:
        taylor.append(taylor_gate_count(ham, time, error, obs_norm))
        trotter.append(trotter_gate_count(ham, time, error))
        qdrift.append(qdrift_gate_count(ham, time, error))

    results = {"taylor": taylor, "trotter": trotter, "qdrift": qdrift}

    for method, config in configs.items():
        result = results[method]
        sns.lineplot(
            x=result,
            y=error_points,
            ax=ax,
            label=config["label"],
            color=config["color"],
        )
        sns.scatterplot(x=result, y=error_points, ax=ax, color=config["color"])

    # SETTING: AXIS VISIBILITY
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"Error ($\log$ scale)")
    ax.set_xlabel(r"Gate Count ($\log$ scale)")

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=3, fontsize=10)
    # plt.legend()
    diagram_name = "plots/benchmark/gate_count.png"
    print(f"Saving diagram at:{diagram_name}")
    plt.savefig(diagram_name, dpi=300)


if __name__ == "__main__":
    qubit = 25
    h_val = 0.1
    start_err_exp = -1
    end_err_exp = -3
    point_count = 10
    obs_norm = 1
    time = 10
    plot_gate_error(
        qubit, h_val, start_err_exp, end_err_exp, point_count, obs_norm, time
    )
