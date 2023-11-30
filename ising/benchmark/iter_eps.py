import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


from ising.hamiltonian.hamiltonian import Hamiltonian
from ising.hamiltonian import parametrized_ising
from ising.benchmark.sim_function import (
    taylor_gate_count,
    trotter_gate_count,
    qdrift_gate_count,
)


def iteration_count(obs_norm: float, delta: float, error: float):
    return (obs_norm / error) ** 2 * np.log(1 / delta)


def plot_iteration_err(start_err_exp, end_err_exp, point_count, obs_norm, delta):
    fig, ax = plt.subplots()

    configs = {
        "taylor": {"color": "blue", "label": "Truncated Taylor"},
        "trotter": {"color": "black", "label": "First Order Trotter"},
        "qdrift": {"color": "red", "label": "qDRIFT Protocol"},
    }

    error_points = [
        10**x for x in np.linspace(start_err_exp, end_err_exp, point_count)
    ]

    result = []
    for error in error_points:
        result.append(iteration_count(obs_norm, delta, error))

    sns.lineplot(x=result, y=error_points, ax=ax, color="red")
    sns.scatterplot(x=result, y=error_points, ax=ax, color="red")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"$\log_{10}(\epsilon)$")
    ax.set_xlabel(r"$\log_{10}(\text{iteration count})$")

    diagram_name = "plots/benchmark/iteration_count.png"
    print(f"Saving diagram at:{diagram_name}")
    plt.savefig(diagram_name, dpi=300)


if __name__ == "__main__":
    start_err_exp = -1
    end_err_exp = -3
    point_count = 10
    obs_norm = 1
    delta = 0.1
    plot_iteration_err(start_err_exp, end_err_exp, point_count, obs_norm, delta)
