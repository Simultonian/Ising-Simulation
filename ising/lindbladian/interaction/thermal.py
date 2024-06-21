import numpy as np


def thermal(eig_vec, eig_val, eig_vec_inv, inv_temp):
    """
    Calculates exp(-betaH) / Tr[exp(-betaH)]
    """
    nr = eig_vec @ np.diag(np.exp(-inv_temp * eig_val)) @ eig_vec_inv
    dr = nr.trace
    return nr / dr
