import numpy as np
from qiskit.quantum_info import SparsePauliOp
from ising.hamiltonian import Hamiltonian


def overall_magnetization(num_qubits) -> Hamiltonian:
    pauli_strings = []
    all_eyes = ["I"] * (num_qubits)

    for i in range(0, num_qubits):
        all_eyes[i] = "Z"
        pauli_i = "".join(all_eyes)
        pauli_strings.append(pauli_i)
        all_eyes[i] = "I"

    coeffs = np.array([1 / num_qubits] * num_qubits)

    return Hamiltonian(
        sparse_repr=SparsePauliOp(pauli_strings, coeffs), normalized=False
    )
