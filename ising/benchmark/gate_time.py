import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ising.hamiltonian import parametrized_ising, trotter_reps, qdrift_count
from ising.benchmark.sim_function import (
    taylor_gate_count,
)


def plot_gate_error(qubit, h_val, error, point_count, obs_norm, start_time, end_time):
    fig, ax = plt.subplots()

    configs = {
        "taylor": {"color": "blue", "label": "Truncated Taylor"},
        "trotter": {"color": "black", "label": "First Order Trotter"},
        "qdrift": {"color": "red", "label": "qDRIFT Protocol"},
    }

    time_points = [x for x in np.linspace(start_time, end_time, point_count)]
    ham = parametrized_ising(qubit, h_val)
    lambd = np.sum(np.abs(ham.coeffs))

    taylor, trotter, qdrift = [], [], []
    for time in time_points:
        taylor.append(taylor_gate_count(ham, time, error, obs_norm))
        trotter.append(trotter_reps(qubit, h_val, time, error))
        qdrift.append(qdrift_count(lambd, time, error))

    results = {"taylor": taylor, "trotter": trotter, "qdrift": qdrift}

    for method, config in configs.items():
        result = results[method]
        sns.lineplot(
            y=result,
            x=time_points,
            ax=ax,
            label=config["label"],
            color=config["color"],
        )

        sns.scatterplot(y=result, x=time_points, ax=ax, color=config["color"])

    # SETTING: AXIS VISIBILITY
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ax.invert_xaxis()
    # ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"time ($t$)")
    ax.set_ylabel(r"Logarithmic Gate Depth")

    # ax.set_title("Depth vs Time for Simulation Techniques")

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=3, fontsize=10)
    # plt.legend()
    diagram_name = "plots/benchmark/gate_time.png"
    print(f"Saving diagram at:{diagram_name}")
    plt.savefig(diagram_name, dpi=300)


if __name__ == "__main__":
    qubit = 6
    h_val = 0.1
    start_time, end_time = 1, 20
    point_count = 10
    obs_norm = 1
    error = 0.1

    plot_gate_error(
        qubit,
        h_val,
        error,
        point_count,
        obs_norm,
        start_time,
        end_time,
    )
