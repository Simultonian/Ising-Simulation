import numpy as np
from itertools import product

from ising.lindbladian.simulation.multi_cm_damping import _kron_multi, apply_amplitude_damping

A, B = np.array([[1, 0], [3, 4]]), np.array([[5, 0], [4, 2]])

def test_kron_prod():
    res = _kron_multi([A, B, A, B])
    expected = np.kron(A, np.kron(B, np.kron(A, B)))
    np.testing.assert_almost_equal(res, expected)


def test_kraus():
    ops = []
    ops.append(np.kron(A, B))
    ops.append(np.kron(A, A))
    ops.append(np.kron(B, B))
    ops.append(np.kron(B, A))

    for kraus_ops in product([A, B], repeat=2):
        term = _kron_multi(kraus_ops)
        assert term in ops
        ops.remove(ops)
    
    assert len(ops) == 0
