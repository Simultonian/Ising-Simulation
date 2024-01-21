import numpy as np
from numpy.typing import NDArray
from qiskit.quantum_info import Pauli

from ising.utils import simdiag


def test_simdiag_simple():
    ps = [Pauli("XI"), Pauli("IX")]
    ms = [np.array(x.to_matrix()) for x in ps]
    e_vals, e_vec = simdiag(ms)
    e_inv = np.linalg.inv(e_vec)

    for ind, m in enumerate(ms):
        e_val = e_vals[ind]
        t_m = e_vec @ np.diag(e_val) @ e_inv
        np.testing.assert_allclose(t_m, m, rtol=1e-7, atol=1e-7)


def test_simdiag_bigger():
    ps = [
        Pauli("ZZIIII"),
        Pauli("IZZIII"),
        Pauli("IIZZII"),
        Pauli("IIIZZI"),
        Pauli("IIIIZZ"),
    ]
    ms = [np.array(x.to_matrix()) for x in ps]
    e_vals, e_vec = simdiag(ms)
    e_inv = np.linalg.inv(e_vec)

    for ind, m in enumerate(ms):
        e_val = e_vals[ind]
        t_m = e_vec @ np.diag(e_val) @ e_inv
        np.testing.assert_allclose(t_m, m, rtol=1e-7, atol=1e-7)


def test_simdiag_x():
    ps = [
        Pauli("XIIIII"),
        Pauli("IXIIII"),
        Pauli("IIXIII"),
        Pauli("IIIXII"),
        Pauli("IIIIXI"),
        Pauli("IIIIIX"),
    ]
    ms = [np.array(x.to_matrix()) for x in ps]
    e_vals, e_vec = simdiag(ms)
    e_inv = np.linalg.inv(e_vec)

    for ind, m in enumerate(ms):
        e_val = e_vals[ind]
        t_m = e_vec @ np.diag(e_val) @ e_inv
        np.testing.assert_allclose(t_m, m, rtol=1e-7, atol=1e-7)


def pauli_matrix(
    eig_val: NDArray, eig_vec: NDArray, eig_inv: NDArray, time: float
) -> NDArray:
    return eig_vec @ np.diag(np.exp(complex(0, -1) * time * eig_val)) @ eig_inv


def test_exp():
    t = 1.0
    ps = [Pauli("XI"), Pauli("IX")]
    ms = [np.array(x.to_matrix()) for x in ps]
    e_vals, e_vec = simdiag(ms)
    e_inv = np.linalg.inv(e_vec)

    final_op = np.eye(2**2)
    for m in ms:
        ea, ee = np.linalg.eig(m)
        ei = np.linalg.inv(ee)
        op = pauli_matrix(ea, ee, ei, t)
        final_op = np.dot(final_op, op)

    e_sum = np.sum(e_vals, axis=0)
    res_op = e_vec @ np.diag(np.exp(complex(0, -1) * t * e_sum)) @ e_inv
    np.testing.assert_almost_equal(res_op, final_op)
