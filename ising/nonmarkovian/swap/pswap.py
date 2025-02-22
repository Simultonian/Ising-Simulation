import numpy as np
from ising.utils import global_phase

SWAP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])
I4 = np.eye(4)


def parameterized_swap(gamma):
    u = I4 * np.cos(gamma) - SWAP * 1j * np.sin(gamma)
    return u / global_phase(u)


def swap_channel(rho, p):
    return (1-p)*rho + p*SWAP @ rho @ SWAP.conj().T