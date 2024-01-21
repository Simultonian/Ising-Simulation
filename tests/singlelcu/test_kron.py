import numpy as np
from ising.utils.constants import ONEONE, ZEROZERO


def kron_control_unitary(unitary, eye, control_val: int):
    """
    Calculates |0><0| U + |1><1| I if control_val = 0
    Calculates |0><0| I + |1><1| U if control_val = 1

    Where U is unitary for Hamiltonian evolution with time `times[ind]`.
    """

    if control_val == 0:
        op_1 = np.kron(ZEROZERO, unitary)
        op_2 = np.kron(ONEONE, eye)
        return op_1 + op_2
    # Control value is 1
    else:
        op_1 = np.kron(ZEROZERO, eye)
        op_2 = np.kron(ONEONE, unitary)
        return op_1 + op_2


def place_control_unitary(unitary, control_val: int):
    """
    Calculates |0><0| U + |1><1| I if control_val = 0
    Calculates |0><0| I + |1><1| U if control_val = 1
    """

    op = np.zeros((2 * unitary.shape[0], 2 * unitary.shape[1]))

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


def test_place_control_unitary():
    # Generate a random matrix
    for _ in range(10):
        control_val = np.random.choice([0, 1])
        a = np.random.rand(3, 3)
        eye = np.identity(a.shape[0])

        # Call the function 'place_control_unitary'
        b = place_control_unitary(a, control_val)

        # Call the function 'kron_control_unitary'
        c = kron_control_unitary(a, eye, control_val)

        # Check if the output of 'place_control_unitary' is the same as the output of 'kron_control_unitary'
        assert np.allclose(b, c)
