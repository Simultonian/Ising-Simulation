from typing import Union
import numpy as np

from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.circuit import QuantumCircuit
from qiskit.synthesis import LieTrotter


class Lie(LieTrotter):
    """The Lie-Trotter product formula."""

    def __init__(self, reps: int = 1) -> None:
        """
        Inputs:
            - reps: The number of time steps.
            - atomic_evolution: A function to construct the circuit for the evolution of single
                Pauli string. Per default, a single Pauli evolution is decomopsed in a CX chain
                and a single qubit Z rotation.
        """
        super().__init__(reps, False, "chain", None)

    def parameterized_map(
        self, operator: Union[list[Pauli], SparsePauliOp], time
    ) -> dict[Pauli, QuantumCircuit]:
        if isinstance(operator, list):
            pauli_list = [(op, 1) for op in operator]
        else:
            pauli_list = [
                (Pauli(op), np.real(coeff)) for op, coeff in operator.to_list()
            ]

        pauli_circuits = {}

        for op, coeff in pauli_list:
            circuit = self.atomic_evolution(op, coeff * time)
            pauli_circuits[op] = circuit

        return pauli_circuits
