import numpy as np
from qiskit.quantum_info import Pauli
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

    def synthesize(self, evolution) -> QuantumCircuit:
        operators = evolution.operator
        time = evolution.time

        evolution_circuit = QuantumCircuit(operators[0].num_qubits)

        if not isinstance(operators, list):
            pauli_list = [
                (Pauli(op), np.real(coeff)) for op, coeff in operators.to_list()
            ]
        else:
            pauli_list = [(op, 1) for op in operators]

        for _ in range(self.reps):
            for op, coeff in pauli_list:
                evolution_circuit.compose(
                    self.atomic_evolution(op, coeff * time / self.reps),
                    wrap=True,
                    inplace=True,
                )

        return evolution_circuit
