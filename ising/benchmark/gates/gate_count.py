import numpy as np
from ising.hamiltonian import Hamiltonian


def trotter_gates(ham: Hamiltonian, time: float, err: float) -> int:
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
    numr = np.log2((lambd * time) * obs_norm / err)
    denr = np.log2(numr)

    return np.ceil(((lambd * time) ** 2) * (numr / denr))
