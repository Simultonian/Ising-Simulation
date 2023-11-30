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
    qubit, h_val, start_err_exp, end_err_exp, point_count, obs_norm, times, scale
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
    ham = parametrized_ising(qubit, h_val)

    linestyle = ["solid", "dashed"]

    for ind, time in enumerate(times):
        taylor, trotter, qdrift = [], [], []
        for error in error_points:
            taylor.append(taylor_gate_count(ham, time, error, obs_norm))
            trotter.append(trotter_gate_count(ham, time, error))
            qdrift.append(qdrift_gate_count(ham, time, error))

        results = {"taylor": taylor, "trotter": trotter, "qdrift": qdrift}

        for method, config in configs.items():
            result = results[method]
            if ind == 0:
                sns.lineplot(
                    x=result,
                    y=error_points,
                    ax=ax,
                    label=config["label"],
                    color=config["color"],
                    linestyle=linestyle[ind],
                )
            else:
                sns.lineplot(
                    x=result,
                    y=error_points,
                    ax=ax,
                    color=config["color"],
                    linestyle=linestyle[ind],
                )

            sns.scatterplot(x=result, y=error_points, ax=ax, color=config["color"])

        # ax.text(error_points[-1] * 2, results["taylor"][-1] * 1.6, f"t={time}")

    # ax.invert_xaxis()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"$\log_{10}(\epsilon)$")
    ax.set_xlabel(r"$\log_{10}(\text{gate count})$")

    ax.set_title("Gate vs Error for Simulation Techniques")

    ax.legend(loc="upper right", framealpha=1)
    # plt.legend()
    diagram_name = "plots/benchmark/gate_count_multi_t.png"
    print(f"Saving diagram at:{diagram_name}")
    plt.savefig(diagram_name, dpi=300)


if __name__ == "__main__":
    qubit = 25
    h_val = 0.1
    start_err_exp = -1
    end_err_exp = -3
    point_count = 10
    obs_norm = 1
    time = [100]
    scale = [1.5, 1.5]
    plot_gate_error(
        qubit, h_val, start_err_exp, end_err_exp, point_count, obs_norm, time, scale
    )
