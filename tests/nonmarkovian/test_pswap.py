import numpy as np
from ising.nonmarkovian.swap import parameterized_swap

SWAP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])

def test_eye():
    res = parameterized_swap(0)
    assert np.allclose(res, np.eye(4))

def test_perfect_swap():
    res = parameterized_swap(np.pi / 2)
    assert np.allclose(res, SWAP)


def test_partial_swap():
    res = parameterized_swap(np.pi / 4)
    assert not np.allclose(res, SWAP)
    assert not np.allclose(res, np.eye(4))
