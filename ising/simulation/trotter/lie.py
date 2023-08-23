from typing import Optional, Union
from functools import lru_cache
import numpy as np
from numpy.typing import NDArray

from qiskit.quantum_info import Pauli, SparsePauliOp, Operator
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.synthesis import LieTrotter

from ising.hamiltonian import Hamiltonian, trotter_reps
from ising.hamiltonian.hamiltonian import substitute_parameter
from ising.utils import MAXSIZE


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

    @property
    def ground_state(self) -> NDArray:
        if self.ham_subbed is None:
            raise ValueError(
                "h value has not been substituted, qiskit does not support parametrized Hamiltonians."
            )
        return self.ham_subbed.ground_state

    def subsitute_h(self, h_val: float) -> None:
        self.ham_subbed = substitute_parameter(self.ham, self.h, h_val)

    def construct_parametrized_circuit(self) -> None:
        if self.ham_subbed is None:
            raise ValueError(
                "h value has not been substituted, qiskit does not support parametrized Hamiltonians."
            )

        self.pauli_mapping = self.synthesizer.parameterized_map(
            self.ham_subbed.sparse_repr, self.time / self.reps
        )

    @lru_cache(maxsize=MAXSIZE)
    def pauli_matrix(self, pauli: Pauli, time: float, reps: int) -> NDArray:
        circuit = self.pauli_mapping[pauli].assign_parameters(
            {self.time: time, self.reps: reps}
        )
        return np.array(Operator.from_circuit(circuit).data).astype(np.complex128)

    def matrix(self, time: float) -> NDArray:
        if self.ham_subbed is None:
            raise ValueError("h value has not been substituted.")
        if self.pauli_mapping is None:
            raise ValueError("Para circuit has not been constructed.")
        reps = trotter_reps(self.ham_subbed.sparse_repr, time, self.error)

        final_op = np.identity(2**self.num_qubits).astype(np.complex128)
        print(f"{time}:{reps}")
        for op in self.ham_subbed.sparse_repr:
            p = self.pauli_matrix(Pauli(op.paulis), time, reps)
            final_op = np.dot(p, final_op)

        return np.linalg.matrix_power(final_op, reps)

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
