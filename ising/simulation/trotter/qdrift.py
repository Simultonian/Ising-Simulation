import numpy as np
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.circuit import QuantumCircuit
from qiskit.synthesis import LieTrotter
from qiskit.circuit.library.pauli_evolution import PauliEvolutionGate

class QDrift(LieTrotter):
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

        if not isinstance(operators, list):
            pauli_list = np.array([Pauli(op) for op, _ in operators.to_list()])
            coeffs = [np.real(coeff) for _, coeff in operators.to_list()]
        else:
            pauli_list = [(op, 1) for op in operators]
            coeffs = [1 for _ in operators]

        weights = np.abs(coeffs)
        lambd = np.sum(weights)

        # num_gates = int(np.ceil(2 * (lambd**2) * (time**2) * self.reps))
        num_gates = int(np.ceil(len(pauli_list) * self.reps))

        evolution_time = lambd * time / num_gates
        pauli_indices = np.array(list(range(len(pauli_list))))
        sampled_indices = np.random.choice(
            pauli_indices,
            size=(num_gates,),
            p=weights / lambd,
        )
        sampled_ops = pauli_list[sampled_indices]
        


        # Build the evolution circuit using the LieTrotter synthesis with the sampled operators
        evolution_circuit = PauliEvolutionGate(
            sum([(SparsePauliOp(op), 1) for op in sampled_ops]),
            time=evolution_time,
            synthesis=LieTrotter(),
        ).definition

        return evolution_circuit
