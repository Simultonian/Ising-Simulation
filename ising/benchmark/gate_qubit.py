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
    time, h_val, error, point_count, obs_norm, start_qubit, end_qubit
):
    fig, ax = plt.subplots()

    configs = {
        "taylor": {"color": "blue", "label": "Truncated Taylor"},
        "trotter": {"color": "black", "label": "First Order Trotter"},
        "qdrift": {"color": "red", "label": "qDRIFT Protocol"},
    }

    qubit_points = [int(x) for x in np.linspace(start_qubit, end_qubit, point_count)]
    taylor, trotter, qdrift = [], [], []

    for qubit in qubit_points:
        ham = parametrized_ising(qubit, h_val)
        taylor.append(taylor_gate_count(ham, time, error, obs_norm))
        trotter.append(trotter_gate_count(ham, time, error))
        qdrift.append(qdrift_gate_count(ham, time, error))

    results = {"taylor": taylor, "trotter": trotter, "qdrift": qdrift}

    for method, config in configs.items():
        result = results[method]
        sns.lineplot(
            y=result,
            x=qubit_points,
            ax=ax,
            label=config["label"],
            color=config["color"],
        )

        sns.scatterplot(y=result, x=qubit_points, ax=ax, color=config["color"])

        # ax.text(error_points[-1] * 2, results["taylor"][-1] * 1.6, f"t={time}")

    # ax.invert_xaxis()
    # ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\text{qubits }(N)$")
    ax.set_ylabel(r"$\log_{10}(\text{gate count})$")

    ax.set_title("Depth vs Qubits for Simulation Techniques")

    ax.legend(loc="upper left", framealpha=1)
    # plt.legend()
    diagram_name = "plots/benchmark/gate_qubits.png"
    print(f"Saving diagram at:{diagram_name}")
    plt.savefig(diagram_name, dpi=300)


if __name__ == "__main__":
    time = 10
    h_val = 1
    start_qubit, end_qubit = 4, 100
    point_count = 10
    obs_norm = 1
    error = 1e-3

    plot_gate_error(
        time, h_val, error, point_count, obs_norm, start_qubit, end_qubit, 
    )
