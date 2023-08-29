from typing import Optional
import numpy as np
from numpy.typing import NDArray

from qiskit.quantum_info import Pauli
from qiskit.circuit import Parameter

from ising.hamiltonian import Hamiltonian, qdrift_count
from ising.hamiltonian import Hamiltonian, trotter_reps, general_grouping
from ising.hamiltonian.hamiltonian import substitute_parameter

from ising.simulation.trotter.grouped_lie import (
    get_grouped_coeffs,
    club_into_groups,
    clubbed_evolve,
    GroupedLie,
)


class GroupedQDriftCircuit:
    def __init__(self, ham: Hamiltonian, h: Parameter, error: float):
        self.ham = ham
        self.num_qubits = ham.sparse_repr.num_qubits
        self.error = error
        self.ham_subbed: Optional[Hamiltonian] = None

        self.h = h
        self.paulis = self.ham.sparse_repr.paulis

        self.synthesizer = GroupedLie(reps=1)
        self.groups = general_grouping(self.ham.sparse_repr.paulis)

        group_map = {}
        for g_ind, group in enumerate(self.groups):
            for p_ind, pauli in enumerate(group):
                group_map[pauli] = (p_ind, g_ind)

        self.group_map = group_map

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
        group_coeffs = [
            np.array(x) for x in get_grouped_coeffs(self.ham_subbed, self.groups)
        ]

        for ind, coeffs in enumerate(group_coeffs):
            for e_ind, (coeff, eig_val) in enumerate(zip(coeffs, self._eigvals[ind])):
                if coeff.real < 0:
                    self.group_mapping[ind][0][e_ind] = -1 * eig_val.real
                else:
                    self.group_mapping[ind][0][e_ind] = eig_val.real

        self.probs = np.abs(self.ham_subbed.sparse_repr.coeffs).astype(np.float64)
        self.lambd = np.sum(self.probs)
        self.probs /= self.lambd
        self.indices = np.array(list(range(len(self.probs))))
        # Use self.ham_subbed.sparse_repr.paulis for accessing paulis

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

        # Sampling Paulis
        count = qdrift_count(self.lambd, time, self.error)
        samples = np.random.choice(self.indices, p=self.probs, size=count)
        paulis = list(self.ham_subbed.sparse_repr.paulis[samples])

        evolution_time = float(self.lambd * time / count)

        # Paulis will be sampled
        club = club_into_groups(paulis, self.group_map)
        final_op = clubbed_evolve(club, self.group_mapping, evolution_time)

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
