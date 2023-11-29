import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ising.hamiltonian import Hamiltonian
from ising.hamiltonian import parametrized_ising


def trotter_gate_count(ham: Hamiltonian, time: float, err: float) -> int:
    """
    First order trotterization gate count for given parameters.
    """
    max_lambd = np.max(np.abs(ham.coeffs))
    l = len(ham.paulis)
    return np.ceil((l**3) * ((max_lambd * time) ** 2) / err)


def qdrift_gate_count(ham: Hamiltonian, time: float, err: float) -> int:
    """
    qDRIFT gate count for given parameters.
    """
    lambd = sum(np.abs(ham.coeffs))
    return np.ceil(((lambd * time) ** 2) / err)


def taylor_gate_count(ham: Hamiltonian, time: float, err: float, obs_norm: int) -> int:
    """
    Truncated Taylor series with single ancilla qubit LCU decomposition.
    """
    lambd = sum(np.abs(ham.coeffs))
    numr = np.log((lambd * time) * obs_norm / err)
    denr = np.log(numr)

    return np.ceil(((lambd * time) ** 2) * (numr / denr))


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
            y=result, x=inv_err, ax=ax, label=config["label"], color=config["color"]
        )
        sns.scatterplot(y=result, x=inv_err, ax=ax, color=config["color"])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$1/\epsilon$")
    ax.set_ylabel(r"$\text{gate count}$")

    plt.legend()
    diagram_name = "plots/benchmark/gate_count.png"
    print(f"Saving diagram at:{diagram_name}")
    plt.savefig(diagram_name)


if __name__ == "__main__":
    qubit = 10
    h_val = 0.1
    start_err_exp = -1
    end_err_exp = -3
    point_count = 10
    obs_norm = 1
    time = 10
    plot_gate_error(
        qubit, h_val, start_err_exp, end_err_exp, point_count, obs_norm, time
    )
