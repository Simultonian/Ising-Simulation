from typing import Optional, Union
import numpy as np
from numpy.typing import NDArray

from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.circuit import Parameter
from qiskit.synthesis import LieTrotter

from ising.hamiltonian import Hamiltonian, trotter_reps, general_grouping
from ising.hamiltonian.hamiltonian import substitute_parameter
from ising.utils import simdiag


class GroupedLie(LieTrotter):
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

    def svd_map(
        self, operator: Union[list[Pauli], SparsePauliOp], groups: list[list[Pauli]]
    ) -> list[tuple[NDArray, NDArray, NDArray]]:
        """Simultaneously diagonalizes the Pauli operators

        Inputs:
        - operator: The operator to be simulated.
        - groups: List of Grouped Pauli operators.

        Returns: List of (eigenvalues, eigenvectors) corresponding to groups
            after adjusting for the coefficients.
        """
        if isinstance(operator, list):
            pauli_map = {Pauli(op): 1 for op in operator}
        else:
            pauli_map = {Pauli(op): np.real(coeff) for op, coeff in operator.to_list()}

        eig_pairs = []
        for group in groups:
            eig_val, eig_vec = simdiag([pauli.to_matrix() for pauli in group])
            for ind, pauli in enumerate(group):
                coeff = pauli_map[pauli]
                eig_val[ind] *= coeff

            eig_inv = np.linalg.inv(eig_vec)

            eig_pairs.append((eig_val, eig_vec, eig_inv))

        return eig_pairs


class GroupedLieCircuit:
    def __init__(self, ham: Hamiltonian, h: Parameter, error: float):
        self.ham = ham
        self.num_qubits = ham.sparse_repr.num_qubits
        self.error = error
        self.ham_subbed: Optional[Hamiltonian] = None

        self.h = h

        self.synthesizer = GroupedLie(reps=1)
        self.groups = general_grouping(self.ham.sparse_repr.paulis)

        group_map = {}
        for g_ind, group in enumerate(self.groups):
            for p_ind, pauli in enumerate(group):
                group_map[pauli] = (p_ind, g_ind)

        self.group_map = group_map

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

        self.group_mapping = self.synthesizer.svd_map(
            self.ham_subbed.sparse_repr, self.groups
        )

    def pauli_matrix(self, pauli: Pauli, time: float, reps: int) -> NDArray:
        p_ind, g_ind = self.group_map[pauli]

        eig_val = self.group_mapping[g_ind][0][p_ind]
        eig_vec = self.group_mapping[g_ind][1]
        eig_inv = self.group_mapping[g_ind][2]

        return (
            eig_vec @ np.diag(np.exp(complex(0, -1) * time / reps * eig_val)) @ eig_inv
        )

    def matrix(self, time: float) -> NDArray:
        """
        Lie Trotter is a deterministic method of generation, using grouping we
        can do the following:
        e^P11 e^P12 ... e^Pmn = V_1 (l_1 + l_2 ...) V_1^t V_2 ... V_2^t ...

        This is faster if the number of groups are less.
        """
        if self.ham_subbed is None:
            raise ValueError("h value has not been substituted.")
        if self.group_mapping is None:
            raise ValueError("Para circuit has not been constructed.")
        reps = trotter_reps(self.ham_subbed.sparse_repr, time, self.error)

        print(f"{time}:{reps}")

        final_op = np.identity(2**self.num_qubits).astype(np.complex128)

        for eig_val, eig_vec, eig_inv in self.group_mapping:
            eig_sum = np.sum(eig_val, axis=0)
            op = (
                eig_vec
                @ np.diag(np.exp(complex(0, -1) * (time / reps) * eig_sum))
                @ eig_inv
            )
            final_op = np.dot(op, final_op)

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
