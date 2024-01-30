from numpy.typing import NDArray
import numpy as np


def apply_kraus(rho: NDArray, kraus: list[NDArray]):
    if len(kraus) == 0:
        return rho

    rho_updated = np.zeros_like(rho)

    for k in kraus:
        rho_updated += k @ rho @ k.conj().T

    return rho_updated
