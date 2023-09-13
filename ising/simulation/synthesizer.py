import numpy as np
from numpy.typing import NDArray

from ising.hamiltonian import Hamiltonian


class Synthesizer:
    def __init__(self, ham: Hamiltonian):
        """
        Base class for Synthesizers
        """
        self.ham = ham
        self.ham_subbed = ham

    def get_unitary(self, t: float) -> NDArray:
        raise ValueError("Base class not to be directly used.")

    def matrix(self, t: float) -> NDArray:
        raise ValueError("Base class not to be directly used.")

    @property
    def ground_state(self) -> NDArray:
        return self.ham_subbed.ground_state

    def get_observations(
        self, rho_init: NDArray, observable: NDArray, times: list[float]
    ):
        raise ValueError("Base class not to be directly used.")
