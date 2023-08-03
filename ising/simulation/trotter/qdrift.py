from typing import Optional
import numpy as np
from numpy.typing import NDArray

from qiskit.quantum_info import Pauli
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import Operator
from qiskit.synthesis import LieTrotter
from qiskit.synthesis.evolution.product_formula import evolve_pauli
from qiskit.circuit.library import PauliEvolutionGate

from ising.hamiltonian import Hamiltonian, trotter_reps
from ising.hamiltonian.hamiltonian import substitute_parameter


class QDrift(LieTrotter):
    """Randomized fast evolution for Hamiltonian Simulation"""

    def __init__(self) -> None:
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


class LieCircuit:
    def __init__(self, ham: Hamiltonian, h: Parameter, error: float):
        self.ham = ham
        self.num_qubits = ham.sparse_repr.num_qubits
        self.error = error
        self.ham_subbed: Optional[Hamiltonian] = None

        self.h = h
        self.time = Parameter("t")

    @property
    def ground_state(self) -> NDArray:
        if self.ham_subbed is None:
            raise ValueError(
                "h value has not been substituted, qiskit does not support parametrized Hamiltonians."
            )
        return self.ham_subbed.ground_state

    def subsitute_h(self, h_val: float) -> None:
        self.ham_subbed = substitute_parameter(self.ham, self.h, h_val)
        self.lambd = sum(self.ham_subbed.sparse_repr.coeffs)

    def construct_parametrized_circuit(self) -> None:
        if self.ham_subbed is None:
            raise ValueError(
                "h value has not been substituted, qiskit does not support parametrized Hamiltonians."
            )

        self.n = Parameter("n")

        self.evo_time = self.lambd * self.time / self.n

        evo_gate1 = PauliEvolutionGate(
            self.ham_subbed.sparse_repr,
            time=self.time / self.reps,
            synthesis=Lie(reps=1),
        )
        circ_h_sub = QuantumCircuit(self.num_qubits)
        circ_h_sub.append(evo_gate1, range(self.num_qubits))
        self.para_circuit = circ_h_sub

    def matrix(self, time: float) -> NDArray:
        if self.ham_subbed is None:
            raise ValueError("h value has not been substituted.")
        if self.para_circuit is None:
            raise ValueError("Para circuit has not been constructed.")
        reps = trotter_reps(self.ham_subbed.sparse_repr, time, self.error)

        circuit = self.para_circuit.assign_parameters(
            {self.reps: reps, self.time: time}
        )
        assert circuit is not None

        return Operator.from_circuit(circuit).power(reps).data

    def get_observations(
        self, rho_init: NDArray, observable: NDArray, times: list[float]
    ):
        results = []
        for time in times:
            n_val = 2 * (self.lambd**2) * (time**2) / self.error
            unitary = self.matrix(time)
            rho_final = unitary @ rho_init @ unitary.conj().T
            result = np.trace(np.abs(observable @ rho_final))
            results.append(result)

        return results
