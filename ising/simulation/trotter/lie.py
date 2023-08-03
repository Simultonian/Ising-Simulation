from typing import Optional
import numpy as np
from numpy.typing import NDArray

from qiskit.quantum_info import Pauli
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import Operator
from qiskit.synthesis import LieTrotter
from qiskit.circuit.library import PauliEvolutionGate

from ising.hamiltonian import Hamiltonian, trotter_reps
from ising.hamiltonian.hamiltonian import substitute_parameter


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

    def synthesize(self, operators, time) -> list[QuantumCircuit]:
        if not isinstance(operators, list):
            pauli_list = [
                (Pauli(op), np.real(coeff)) for op, coeff in operators.to_list()
            ]
        else:
            pauli_list = [(op, 1) for op in operators]

        operations_list = []
        for _ in range(self.reps):
            for op, coeff in pauli_list:
                operations_list.append(self.atomic_evolution(op, coeff * time / self.reps))

        return operations_list


class LieCircuit:
    def __init__(self, ham: Hamiltonian, h: Parameter, error: float):
        self.ham = ham
        self.num_qubits = ham.sparse_repr.num_qubits
        self.error = error
        self.ham_subbed: Optional[Hamiltonian] = None

        self.h = h
        self.time = Parameter("t")
        self.reps = Parameter("r")
        
        self.synthesizer = Lie(reps=1)
        self.para_circuits = []

    @property
    def ground_state(self) -> NDArray:
        if self.ham_subbed is None:
            raise ValueError(
                "h value has not been substituted, qiskit does not support parametrized Hamiltonians."
            )
        return self.ham_subbed.ground_state

    def subsitute_h(self, h_val: float) -> None:
        self.ham_subbed = substitute_parameter(self.ham, self.h, h_val)


    def fast_pauli(self, index: int, reps:int, time: float) -> NDArray:
        circ = self.para_circuits[index].assign_parameters({self.reps: reps, self.time: time})
        return Operator.from_circuit(circ).data

    def construct_parametrized_circuit(self) -> None:
        if self.ham_subbed is None:
            raise ValueError(
                "h value has not been substituted, qiskit does not support parametrized Hamiltonians."
            )

        self.para_circuits = self.synthesizer.synthesize(self.ham_subbed.sparse_repr, self.time/self.reps)

    def matrix(self, time: float) -> NDArray:
        if self.ham_subbed is None:
            raise ValueError("h value has not been substituted.")
        if len(self.para_circuits) == 0: 
            raise ValueError("Para circuit has not been constructed.")
        reps = trotter_reps(self.ham_subbed.sparse_repr, time, self.error)

        final_answer = self.fast_pauli(0, reps, time)
        for ind in range(1, len(self.para_circuits)):
            final_answer = self.fast_pauli(ind, reps, time) @ final_answer 

        return np.power(final_answer, reps)

    def get_observations(
        self, rho_init: NDArray, observable: NDArray, times: list[float]
    ):
        results = []
        for time in times:
            unitary = self.matrix(time)
            rho_final = unitary @ rho_init @ unitary.conj().T
            result = np.trace(np.abs(observable @ rho_final))
            results.append(result)

        return results
