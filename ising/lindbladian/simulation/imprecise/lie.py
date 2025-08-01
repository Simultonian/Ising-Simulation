from typing import Optional, Union, Sequence
import numpy as np
from numpy.typing import NDArray

from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.circuit import QuantumCircuit
from qiskit.synthesis import LieTrotter

from ising.hamiltonian import Hamiltonian, general_grouping
from ising.hamiltonian.ising_one import trotter_reps_general

from ising.utils import control_version
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

    def svd_map(self, groups: list[list[Pauli]]) -> list[list[NDArray]]:
        """Simultaneously diagonalizes the Pauli operators

        Inputs:
        - operator: The operator to be simulated.
        - groups: List of Grouped Pauli operators.

        Returns: List of (eigenvalues, eigenvectors) corresponding to groups
            after adjusting for the coefficients.
        """
        eig_pairs = []
        for group in groups:
            mats = [pauli.to_matrix() for pauli in group]

            eig_val, eig_vec = simdiag([pauli.to_matrix() for pauli in group])
            eig_inv = np.linalg.inv(eig_vec)

            for ind, mat in enumerate(mats):
                ev = eig_val[ind]
                res = eig_vec @ np.diag(ev) @ eig_inv
                np.testing.assert_allclose(mat, res, rtol=1e-7, atol=1e-9)

            eig_pairs.append([eig_val, eig_vec, eig_inv])

        return eig_pairs


def get_grouped_coeffs(
    ham: Hamiltonian, groups: list[list[Pauli]]
) -> list[list[complex]]:
    return [[ham[pauli] for pauli in group] for group in groups]


def club_into_groups(
    paulis: Sequence[int], group_map: dict[int, tuple[int, int]]
) -> list[tuple[int, tuple[int, ...]]]:
    """
    Clubs the list of pauli indices into adjacent groups. The return value is
    of the form: [(group, [paulis])*]
    """
    grouped = []
    cur_group = -1
    inds = []
    for pauli in paulis:
        p, g = group_map[pauli]

        if cur_group == -1:
            cur_group = g

        if cur_group == g:
            inds.append(p)
            continue
        # New group
        assert len(inds) > 0
        inds.sort()
        grouped.append((cur_group, tuple(inds)))
        cur_group = g
        inds = [p]

    if len(inds) > 0 and cur_group != -1:
        inds.sort()
        grouped.append((cur_group, tuple(inds)))

    return grouped


def clubbed_evolve(
    club: list[tuple[int, tuple[int, ...]]],
    group_mapping: list[list[NDArray[np.complex128]]],
    time: float,
) -> NDArray:
    """
    Takes in the clubbed operators and constructs the matrices using the
    provided decomposition.
    """

    # Final shape is identitcal to the eigenvector matrix
    final_op = np.identity(len(group_mapping[0][1]))

    for group, paulis in club:
        eig_val, eig_vec, eig_inv = group_mapping[group]
        eig_sum = eig_val.take(paulis, axis=0).sum(axis=0)
        op = eig_vec @ np.diag(np.exp(complex(0, -1) * time * eig_sum)) @ eig_inv
        final_op = np.dot(op, final_op)

    return final_op


class GroupedLieCircuit:
    def __init__(self, ham: Hamiltonian, error: float):
        self.ham = ham
        self.num_qubits = ham.sparse_repr.num_qubits
        self.error = error
        self.obs: Optional[NDArray] = None

        self.synthesizer = GroupedLie(reps=1)
        self.groups = general_grouping(self.ham.sparse_repr.paulis)

        group_map = {}
        for g_ind, group in enumerate(self.groups):
            for p_ind, pauli in enumerate(group):
                group_map[pauli] = (p_ind, g_ind)

        self.group_map = group_map

        self.svd_map = self.synthesizer.svd_map(self.groups)
        self._eigvals = [np.copy(x[0]) for x in self.svd_map]


        self.group_coeffs = [
            np.array(x) for x in get_grouped_coeffs(self.ham, self.groups)
        ]

        for g_ind, coeffs in enumerate(self.group_coeffs):
            for e_ind, (coeff, eig_val) in enumerate(zip(coeffs, self._eigvals[g_ind])):
                self.svd_map[g_ind][0][e_ind] = coeff.real * eig_val

        self.eig_sums = [
            np.sum(self.svd_map[g_ind][0], axis=0)
            for g_ind, _ in enumerate(self.groups)
        ]

    @property
    def ground_state(self) -> NDArray:
        return self.ham.ground_state

    def pauli_matrix(self, pauli: Pauli, time: float, reps: int) -> NDArray:
        p_ind, g_ind = self.group_map[pauli]
        eig_val = self.svd_map[g_ind][0][p_ind]
        eig_vec = self.svd_map[g_ind][1]
        eig_inv = self.svd_map[g_ind][2]

        assert reps > 0
        return (
            eig_vec
            @ np.diag(np.exp(complex(0, -1) * (time / reps) * eig_val))
            @ eig_inv
        )

    def group_matrix(self, g_ind: int, time: float, reps: int) -> NDArray:
        eig_sum = self.eig_sums[g_ind]
        eig_vec = self.svd_map[g_ind][1]
        eig_inv = self.svd_map[g_ind][2]

        assert reps > 0
        return (
            eig_vec
            @ np.diag(np.exp(complex(0, -1) * (time / reps) * eig_sum))
            @ eig_inv
        )

    def matrix(self, time: float, reps: int = -1) -> NDArray:
        if reps == -1:
            reps = trotter_reps_general(ham = self.ham.sparse_repr, time = time, eps = self.error)
 
        final_op = np.identity(2**self.num_qubits).astype(np.complex128)

        for g_ind, _ in enumerate(self.groups):
            group_op = self.group_matrix(g_ind, time, reps)
            final_op = np.dot(group_op, final_op)

        return np.linalg.matrix_power(final_op, reps)
