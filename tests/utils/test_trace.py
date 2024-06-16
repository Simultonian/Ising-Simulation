import numpy as np
from ising.utils.trace import partial_trace
from qiskit.quantum_info import partial_trace as qiskit_partial_trace, DensityMatrix

ZERO = [[1], [0]]
ONE = [[0], [1]]


def test_oneone():
    oneone = np.array([0, 0, 0, 1])
    rho_oneone = np.outer(oneone, oneone.conj())

    rho_2 = partial_trace(rho_oneone, [])

    expected = np.array([[0, 0], [0, 1]])
    np.testing.assert_almost_equal(rho_2, expected)

    expected = qiskit_partial_trace(DensityMatrix(rho_oneone), [0])
    np.testing.assert_almost_equal(rho_2, expected)

    expected = qiskit_partial_trace(rho_oneone, [1])
    np.testing.assert_almost_equal(rho_2, expected)
