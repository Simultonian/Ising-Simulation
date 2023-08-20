from typing import Optional, Union
from functools import lru_cache
from scipy import sparse
import numpy as np
from numpy.typing import NDArray

from qiskit.quantum_info import Pauli
from qiskit.circuit import Parameter

from ising.hamiltonian import Hamiltonian, trotter_reps
from ising.utils import MAXSIZE
from lie import LieCircuit


class SparseLie(LieCircuit):
    def __init__(self, ham: Hamiltonian, h: Parameter, error: float):
        super().__init__(ham, h, error)

    @lru_cache(maxsize=MAXSIZE)
    def pauli_matrix(self, pauli: Pauli, time: float, reps: int) -> sparse.bsr_array:
        return sparse.bsr_array(super().pauli_matrix(pauli, time, reps))

    def matrix(self, time: float) -> NDArray:
        if self.ham_subbed is None:
            raise ValueError("h value has not been substituted.")
        if self.pauli_mapping is None:
            raise ValueError("Para circuit has not been constructed.")
        reps = trotter_reps(self.ham_subbed.sparse_repr, time, self.error)

        final_op = sparse.bsr_array(np.identity(2**self.num_qubits))
        print(f"{time}:{reps}")
        for op in self.ham_subbed.sparse_repr:
            p = self.pauli_matrix(Pauli(op.paulis), time, reps)
            final_op = p.dot(final_op)

        return np.linalg.matrix_power(final_op.todense(), reps)
