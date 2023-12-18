from typing import Optional
from functools import lru_cache
import numpy as np
from numpy.typing import NDArray

from qiskit.quantum_info import Pauli
from qiskit.circuit import Parameter

from ising.hamiltonian import Hamiltonian, general_grouping
from ising.hamiltonian import Hamiltonian, general_grouping, qdrift_count
from ising.hamiltonian.hamiltonian import substitute_parameter
from ising.utils import MAXSIZE, control_version

from ising.simulation.trotter import GroupedLie
from ising.simulation.trotter.grouped_lie import (
    get_grouped_coeffs,
)
from ising.simulation.trotter.grouped_lie import get_grouped_coeffs


def circuit_depth(lambd: float, time: float, err: float) -> int:
    """
    Gets the number of iterations required to get the value to epsilon close.
    """
    reps = qdrift_count(lambd, time, err)
    return reps


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
        self.num_qubits = ham.sparse_repr.num_qubits
        self.error = error
        self.ham_subbed: Optional[Hamiltonian] = None

        self.h = h
        self.time = Parameter("t")
        self.reps = Parameter("r")

        self.synthesizer = GroupedLie(reps=1)
        self.groups = general_grouping(self.ham.sparse_repr.paulis)

        group_map = {}
        for g_ind, group in enumerate(self.groups):
            for p_ind, pauli in enumerate(group):
                group_map[pauli] = (p_ind, g_ind)

        self.group_map = group_map

        self.group_mapping = self.synthesizer.svd_map(self.groups)
        self._eigvals = [x[0] for x in self.group_mapping]
        self.svd_map = self.synthesizer.svd_map(self.groups)
        self._eigvals = [np.copy(x[0]) for x in self.svd_map]

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

        self.pauli_mapping = self.synthesizer.parameterized_map(
            self.ham_subbed.sparse_repr, self.time / self.reps
        )

    def substitute_obs(self, obs: Hamiltonian):
        self.obs = obs.matrix

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

    @lru_cache(maxsize=MAXSIZE)
    def group_matrix(self, g_ind: int, time: float, reps: int) -> NDArray:
        eig_sum = self.eig_sums[g_ind]
        eig_vec = self.svd_map[g_ind][1]
        eig_inv = self.svd_map[g_ind][2]

        return (
            eig_vec
            @ np.diag(np.exp(complex(0, -1) * (time / reps) * eig_sum))
            @ eig_inv
        )

    @lru_cache(maxsize=MAXSIZE)
    def club_matrix(self, club: tuple[int, int], time: float, reps: int) -> NDArray:
        g_ind, count = club
        single = self.group_matrix(g_ind, time, reps)
        return np.linalg.matrix_power(single, count)

    def matrix(self, time: float) -> NDArray:
        if self.ham_subbed is None:
            raise ValueError("h value has not been substituted.")
        if self.pauli_mapping is None:
            raise ValueError("Para circuit has not been constructed.")

        # Sampling
        count = qdrift_count(self.lambd, time, self.error)
        samples = np.random.choice(self.indices, p=self.probs, size=count).astype(int)

        clubs = club_same_terms(samples)

        final_op = np.identity(2**self.num_qubits).astype(np.complex128)

        for club in clubs:
            group_op = self.club_matrix(club, self.lambd * time, count)
            final_op = np.dot(group_op, final_op)

        return final_op

    def evolve(self, psi_init: NDArray, time) -> NDArray:
        return self.matrix(time) @ psi_init

    def control_evolve(
        self, psi_init: NDArray, time: float, control_val: int
    ) -> NDArray:
        op = control_version(self.matrix(time), control_val)
        return op @ psi_init

    def get_observations(self, psi_init: NDArray, times: list[float]):
        if self.obs is None:
            raise ValueError("Observable not set")

        results = []
        for time in times:
            depth = qdrift_count(self.lambd, time, self.error)
            print(f"Time: {time} Depth: {depth}")

            final_psi = self.evolve(psi_init, time)
            final_rho = np.outer(final_psi, final_psi.conj())
            result = np.trace(np.abs(self.obs @ final_rho))
            results.append(result)

        return results
