import numpy as np
from qiskit.quantum_info import partial_trace as qiskit_partial_trace


def partial_trace(rho, indexes):
    siz = int(np.log2(rho.shape[0]))
    # reverse
    new_indexes = [(siz - 1) - ind for ind in indexes]
    return qiskit_partial_trace(rho, new_indexes).data
