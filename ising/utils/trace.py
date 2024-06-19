import numpy as np


def partial_trace(mat, pos):
    """
    Take partial trace of all the Hilbert spaces present at pos (0-indexed)
    and return the matrix
    """
    reshaped_dm = mat.reshape([2, 2, 2, 2])
    # partial trace the second space
    reduced_dm = np.einsum("ijkl->jk", reshaped_dm)
    return reduced_dm
