import numpy as np
from ising.hamiltonian import Hamiltonian
from numpy.typing import NDArray
from ising.utils.trace import partial_trace


def matrix_exp(eig_vec, eig_val, eig_vec_inv, time: float):
    return eig_vec @ np.diag(np.exp(complex(0, -1) * time * eig_val)) @ eig_vec_inv


def set_ham_into_pos(ham, l_size, r_size):
    """
    Given `Ham` set it into a larger Hilbert space with other portions being
    identity.

    Inputs:
        - ham: Input Hamiltonian
        - l_size: number of Identity operators on left
        - r_size: number of Identity operators on right

    Returns: Final Hamiltonian
    """
    l_eye = np.eye(2**l_size)
    r_eye = np.eye(2**r_size)
    return np.kron(l_eye, np.kron(ham, r_eye))


class RepetitionMap:
    def __init__(
        self,
        sys_ham: NDArray,
        env_hams: list[NDArray],
        interaction_hams: list[NDArray],
        interaction_strength: float,
    ):
        """
        This class captures the RI map dynamics and the functionalities of the
        same.
        """

        if len(interaction_hams) != len(env_hams):
            raise ValueError(
                "Size mismatch: interaction hams and env hams differ in size."
            )

        self.sys_ham = sys_ham
        self.env_hams = env_hams
        self.interaction_hams = interaction_hams
        self.interaction_strength = interaction_strength

        self.h_terms = []

        for index in range(len(self.interaction_hams)):
            total_ham = (
                self.sys_ham
                + self.env_hams[index]
                + (self.interaction_strength * self.interaction_hams[index])
            )
            ham_val, ham_vec = np.linalg.eig(total_ham)
            ham_inv = np.linalg.inv(ham_vec)

            self.h_terms.append((ham_vec, ham_val, ham_inv))

    def apply_ri(self, sys_state: NDArray, env_state: NDArray, tao: float, index: int):
        """
        Apply the map by \tao to the initial state and return the state after
        tracing out the state.

        """

        if index >= len(self.interaction_hams):
            raise ValueError("index out of range")

        init_state = np.kron(sys_state, env_state)

        eig_vec, eig_val, eig_inv = self.h_terms[index]

        # e^{-iHt} \rho e^{iHt}
        fin_state = (
            matrix_exp(eig_vec, eig_val, eig_inv, tao)
            @ init_state
            @ matrix_exp(eig_vec, eig_val, eig_inv, -tao)
        )

        sys_size = int(np.log2(sys_state.shape[0]))
        comp_size = int(np.log2(init_state.shape[0]))

        # Tracing out other
        rho_sys = partial_trace(fin_state, list(range(sys_size, comp_size)))
        return rho_sys
