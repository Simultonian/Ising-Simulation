import numpy as np
from ising.utils.trace import partial_trace
from qiskit.quantum_info import partial_trace as qiskit_partial_trace, DensityMatrix

ZERO = [[1], [0]]
ONE = [[0], [1]]
BASIS = [ZERO, ONE]


def test_oneone():
    # |11>
    oneone = np.array([0, 0, 0, 1])
    rho_oneone = np.outer(oneone, oneone.conj())

    rho_2 = partial_trace(rho_oneone, [1])

    expected = np.array([[0, 0], [0, 1]])
    np.testing.assert_almost_equal(rho_2, expected)

    expected = qiskit_partial_trace(DensityMatrix(rho_oneone), [0])
    np.testing.assert_almost_equal(rho_2, expected)

    expected = qiskit_partial_trace(rho_oneone, [1])
    np.testing.assert_almost_equal(rho_2, expected)

def test_basis_three_qubits():
    for psi0 in BASIS:
        for psi1 in BASIS:
            for psi2 in BASIS:
                psi = np.kron(np.kron(psi0, psi1), psi2)
                rho = np.outer(psi, psi.conj())

                psi_01 = np.kron(psi0, psi1)
                rho_01 = np.outer(psi_01, psi_01.conj())

                psi_02 = np.kron(psi0, psi2)
                rho_02 = np.outer(psi_02, psi_02.conj())

                psi_12 = np.kron(psi1, psi2)
                rho_12 = np.outer(psi_12, psi_12.conj())

                exp_01 = partial_trace(rho, [2])
                exp_02 = partial_trace(rho, [1])
                exp_12 = partial_trace(rho, [0])

                np.testing.assert_almost_equal(rho_02, exp_02)
                np.testing.assert_almost_equal(rho_01, exp_01)
                np.testing.assert_almost_equal(rho_12, exp_12)
