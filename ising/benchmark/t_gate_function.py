import numpy as np
from ising.hamiltonian import Hamiltonian


def trotter_gate_count(ham: Hamiltonian, time: float, err: float) -> int:
    """
    First order trotterization gate count for given parameters.
    """
    h = np.min(np.abs(ham.coeffs))
    l = len(ham.paulis)
    r = np.ceil((h * ham.num_qubits * (time**2)) / err)
    return l * r


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
    numr = np.log2((lambd * time) * obs_norm / err)
    denr = np.log2(numr)

    return np.ceil(((lambd * time) ** 2) * (numr / denr))
