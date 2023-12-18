import numpy as np


def control_version(unitary, control_val: int):
    """
    Calculates |0><0| U + |1><1| I if control_val = 0
    Calculates |0><0| I + |1><1| U if control_val = 1
    """

    op = np.zeros((2 * unitary.shape[0], 2 * unitary.shape[1]), dtype=complex)

    if control_val == 0:
        # Top left corner
        op[: unitary.shape[0], : unitary.shape[1]] = unitary

        # Bottom right corner
        np.fill_diagonal(op[unitary.shape[0] :, unitary.shape[1] :], 1)

    else:
        # Top left corner
        np.fill_diagonal(op[: unitary.shape[0], : unitary.shape[1]], 1)

        # Bottom right corner
        op[unitary.shape[0] :, unitary.shape[1] :] = unitary

    return op
