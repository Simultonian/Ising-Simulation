from typing import Optional
from functools import lru_cache
import numpy as np
from numpy.typing import NDArray

from qiskit.quantum_info import Pauli
from qiskit.circuit import Parameter

from ising.hamiltonian import Hamiltonian, trotter_reps, general_grouping
from ising.hamiltonian import Hamiltonian, general_grouping, qdrift_count
from ising.hamiltonian.hamiltonian import substitute_parameter
from ising.utils import MAXSIZE

from ising.simulation.trotter import GroupedLie
from ising.simulation.trotter.grouped_lie import (
    get_grouped_coeffs,
    club_into_groups,
    clubbed_evolve,
)
from ising.simulation.trotter.grouped_lie import get_grouped_coeffs


def club_same_terms(terms: NDArray[np.int64]) -> list[tuple[int, int]]:
    """
    If QDrift samples [0, 0, 1, 1, 1, 0] group then this function clubs and
    returns [(0, 2), (1, 3), (0, 1)]
    """
    club = []
    cur = terms[0]
    count = 1
    for term in terms[1:]:
        if cur == term:
            count += 1
        else:
            club.append((cur, count))
            cur = term
            count = 1

    club.append((cur, count))
    return club


class GSQDriftCircuit:
    def __init__(self, ham: Hamiltonian, h: Parameter, error: float):
        self.ham = ham
        self.error = error
        self.ham_subbed: Optional[Hamiltonian] = None

        self.h = h
        self.time = Parameter("t")
        self.reps = Parameter("r")

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
            for p_ind, pauli in enumerate(group):
                group_map[pauli] = (p_ind, g_ind)

        self.group_map = group_map
        self.order = club_into_groups(inds, group_map)

        self.group_mapping = self.synthesizer.svd_map(self.groups)
        self._eigvals = [x[0] for x in self.group_mapping]
        self.svd_map = self.synthesizer.svd_map(self.groups)
        self._eigvals = [np.copy(x[0]) for x in self.svd_map]

    @property
    def ground_state(self) -> NDArray:
@ -52,13 +68,30 @@ class GSQDriftCircuit:

    def subsitute_h(self, h_val: float) -> None:
        self.ham_subbed = substitute_parameter(self.ham, self.h, h_val)

        self.group_coeffs = [
            np.array(x) for x in get_grouped_coeffs(self.ham_subbed, self.groups)
        ]

        for ind, coeffs in enumerate(self.group_coeffs):
            for e_ind, (coeff, eig_val) in enumerate(zip(coeffs, self._eigvals[ind])):
                self.group_mapping[ind][0][e_ind] = coeff.real * eig_val.real
        group_totals = []
        for g_ind, coeffs in enumerate(self.group_coeffs):
            group_total = np.sum(np.abs(coeffs))
            group_totals.append(group_total)

            if group_total == 0:
                continue

            for e_ind, (coeff, eig_val) in enumerate(zip(coeffs, self._eigvals[g_ind])):
                self.svd_map[g_ind][0][e_ind] = eig_val * (coeff.real / group_total)

        self.lambd = np.sum(group_totals).astype(float)
        self.probs = np.array(group_totals, dtype=float) / self.lambd
        self.indices = np.arange(len(self.probs))

        self.eig_sums = [
            np.sum(self.svd_map[g_ind][0], axis=0)
            for g_ind, _ in enumerate(self.groups)
        ]

    def construct_parametrized_circuit(self) -> None:
        if self.ham_subbed is None:
@ -66,36 +99,61 @@ class GSQDriftCircuit:
                "h value has not been substituted, qiskit does not support parametrized Hamiltonians."
            )

        self.pauli_mapping = self.synthesizer.parameterized_map(
            self.ham_subbed.sparse_repr, self.time / self.reps
        )

    @lru_cache(maxsize=MAXSIZE)
    def pauli_matrix(self, pauli: Pauli, time: float, reps: int) -> NDArray:
        p_ind, g_ind = self.group_map[pauli]
        eig_val = self.svd_map[g_ind][0][p_ind]
        eig_vec = self.svd_map[g_ind][1]
        eig_inv = self.svd_map[g_ind][2]

        return (
            eig_vec
            @ np.diag(np.exp(complex(0, -1) * (time / reps) * eig_val))
            @ eig_inv
        )

        eig_val = self.group_mapping[g_ind][0][p_ind]
        eig_vec = self.group_mapping[g_ind][1]
        eig_inv = self.group_mapping[g_ind][2]
    @lru_cache(maxsize=MAXSIZE)
    def group_matrix(self, g_ind: int, time: float, reps: int) -> NDArray:
        eig_sum = self.eig_sums[g_ind]
        eig_vec = self.svd_map[g_ind][1]
        eig_inv = self.svd_map[g_ind][2]

        return (
            eig_vec @ np.diag(np.exp(complex(0, -1) * time / reps * eig_val)) @ eig_inv
            eig_vec
            @ np.diag(np.exp(complex(0, -1) * (time / reps) * eig_sum))
            @ eig_inv
        )

    def matrix(self, time: float) -> NDArray:
        """
        Lie Trotter is a deterministic method of generation, using grouping we
        can do the following:
        e^P11 e^P12 ... e^Pmn = V_1 (l_1 + l_2 ...) V_1^t V_2 ... V_2^t ...
    @lru_cache(maxsize=MAXSIZE)
    def club_matrix(self, club: tuple[int, int], time: float, reps: int) -> NDArray:
        g_ind, count = club
        single = self.group_matrix(g_ind, time, reps)
        return np.linalg.matrix_power(single, count)

        This is faster if the number of groups are less.
        """
    def matrix(self, time: float) -> NDArray:
        if self.ham_subbed is None:
            raise ValueError("h value has not been substituted.")
        if self.group_mapping is None:
        if self.pauli_mapping is None:
            raise ValueError("Para circuit has not been constructed.")
        reps = trotter_reps(self.ham_subbed.sparse_repr, time, self.error)

        print(f"{time}:{reps}")
        # Sampling
        count = qdrift_count(self.lambd, time, self.error)
        samples = np.random.choice(self.indices, p=self.probs, size=count).astype(int)

        clubs = club_same_terms(samples)

        final_op = np.identity(2**self.num_qubits).astype(np.complex128)
        print(f"{time}:{count}")

        final_op = clubbed_evolve(self.order, self.group_mapping, time / reps)
        for club in clubs:
            group_op = self.club_matrix(club, self.lambd * time, count)
            final_op = np.dot(group_op, final_op)

        return np.linalg.matrix_power(final_op, reps)
        return final_op

    def get_observations(
        self, rho_init: NDArray, observable: NDArray, times: list[float]
