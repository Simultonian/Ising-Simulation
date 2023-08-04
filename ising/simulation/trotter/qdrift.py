from typing import Optional
from functools import lru_cache
import numpy as np
from numpy.typing import NDArray

from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import Operator
from qiskit.synthesis import LieTrotter, QDrift
from qiskit.circuit.library import PauliEvolutionGate
from ising.hamiltonian import Hamiltonian, qdrift_count
from ising.hamiltonian.hamiltonian import substitute_parameter
from ising.simulation.trotter import Lie


class QDriftCircuit:
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
        self.coeffs = []
        self.paulis = []
        self.sign_map = {}

    def construct_parametrized_circuit(self) -> None:
        if self.ham_subbed is None:
            raise ValueError(
                "h value has not been substituted, qiskit does not support parametrized Hamiltonians."
            )

        coeffs = []
        paulis = []
        sign_map = {}
        for op, coeff in self.ham_subbed.sparse_repr.to_list():
            if coeff.real < 0:
                sign_map[Pauli(op)] = -1
                coeff = -coeff
            else:
                sign_map[Pauli(op)] = 1

            coeffs.append(coeff.real)
            paulis.append(Pauli(op))

        self.coeffs = np.abs(np.array(coeffs))
        self.lambd = sum(self.coeffs)
        self.paulis = paulis
        self.sign_map = sign_map
        self.pauli_mapping = self.synthesizer.parameterized_map(self.paulis, self.time)

    @lru_cache
    def pauli_matrix(self, pauli: Pauli, time: float) -> NDArray:
        circuit = self.pauli_mapping[pauli].assign_parameters(
            {self.time: self.sign_map[pauli] * time}
        )
        return Operator.from_circuit(circuit).reverse_qargs().data

    def matrix(self, time: float) -> NDArray:
        if self.ham_subbed is None:
            raise ValueError("h value has not been substituted.")
        if len(self.pauli_mapping) == 0:
            raise ValueError("Para circuit has not been constructed.")

        count = qdrift_count(self.lambd, time, self.error)
        evolution_time = self.lambd * time / count

        final_op = np.identity(2**self.num_qubits)
        indices = list(range(len(self.paulis)))

        samples = np.random.choice(indices, p=self.coeffs / self.lambd, size=count)
        for sample in samples:
            p = self.pauli_matrix(self.paulis[sample], evolution_time)
            final_op = np.dot(p, final_op)

        return final_op

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
