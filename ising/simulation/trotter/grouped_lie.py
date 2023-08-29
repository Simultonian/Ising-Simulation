from typing import Optional, Sequence
import numpy as np
from numpy.typing import NDArray

from qiskit.quantum_info import Pauli
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
            eig_val, eig_vec = simdiag([pauli.to_matrix() for pauli in group])
            eig_inv = np.linalg.inv(eig_vec)

            eig_pairs.append([eig_val, eig_vec, eig_inv])

        return eig_pairs


def get_grouped_coeffs(
    ham: Hamiltonian, groups: list[list[Pauli]]
) -> list[list[complex]]:
    return [[ham[pauli] for pauli in group] for group in groups]


def club_into_groups(
    paulis: Sequence[int], group_map: dict[int, tuple[int, int]]
) -> list[tuple[int, list[int]]]:
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
        grouped.append((cur_group, inds))
        cur_group = g
        inds = [p]

    if len(inds) > 0 and cur_group != -1:
        inds.sort()
        grouped.append((cur_group, inds))

    return grouped


def clubbed_evolve(
    club: list[tuple[int, list[int]]],
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
        eig_sum = eig_val[paulis].sum(axis=0)
        op = eig_vec @ np.diag(np.exp(complex(0, -1) * time * eig_sum)) @ eig_inv
        final_op = np.dot(op, final_op)

    return final_op


class GroupedLieCircuit:
    def __init__(self, ham: Hamiltonian, h: Parameter, error: float):
        self.ham = ham
        self.num_qubits = ham.sparse_repr.num_qubits
        self.error = error
        self.ham_subbed: Optional[Hamiltonian] = None

        self.h = h

        self.synthesizer = GroupedLie(reps=1)
        self.groups = general_grouping(self.ham.sparse_repr.paulis)

        inds = []
        ind_count = 0
        group_map = {}
        for g_ind, group in enumerate(self.groups):
            for p_ind, _ in enumerate(group):
                group_map[ind_count] = (p_ind, g_ind)
                inds.append(ind_count)
                ind_count += 1

        self.group_map = group_map
        self.order = club_into_groups(inds, group_map)

        self.group_mapping = self.synthesizer.svd_map(self.groups)
        self._eigvals = [x[0] for x in self.group_mapping]

    @property
    def ground_state(self) -> NDArray:
        if self.ham_subbed is None:
            raise ValueError(
                "h value has not been substituted, qiskit does not support parametrized Hamiltonians."
            )
        return self.ham_subbed.ground_state

    def subsitute_h(self, h_val: float) -> None:
        self.ham_subbed = substitute_parameter(self.ham, self.h, h_val)
        self.group_coeffs = [
            np.array(x) for x in get_grouped_coeffs(self.ham_subbed, self.groups)
        ]

        for ind, coeffs in enumerate(self.group_coeffs):
            for e_ind, (coeff, eig_val) in enumerate(zip(coeffs, self._eigvals[ind])):
                self.group_mapping[ind][0][e_ind] = coeff.real * eig_val.real

    def construct_parametrized_circuit(self) -> None:
        if self.ham_subbed is None:
            raise ValueError(
                "h value has not been substituted, qiskit does not support parametrized Hamiltonians."
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

        final_op = clubbed_evolve(self.order, self.group_mapping, time / reps)

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
