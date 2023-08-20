import numpy as np
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
    ps = [Pauli("ZZIIII"), Pauli("IZZIII"), Pauli("IIZZII"), Pauli("IIIZZI"), Pauli("IIIIZZ")]
    ms = [np.array(x.to_matrix()) for x in ps]
    e_vals, e_vec = simdiag(ms)
    e_inv = np.linalg.inv(e_vec)

    for ind, m in enumerate(ms):
        e_val = e_vals[ind]
        t_m = e_vec @ np.diag(e_val) @ e_inv
        np.testing.assert_allclose(t_m, m, rtol=1e-7, atol=1e-7)

def test_simdiag_x():
    ps = [Pauli("XIIIII"), Pauli("IXIIII"), Pauli("IIXIII"), Pauli("IIIXII"), Pauli("IIIIXI"), Pauli("IIIIIX")]
    ms = [np.array(x.to_matrix()) for x in ps]
    e_vals, e_vec = simdiag(ms)
    e_inv = np.linalg.inv(e_vec)

    for ind, m in enumerate(ms):
        e_val = e_vals[ind]
        t_m = e_vec @ np.diag(e_val) @ e_inv
        np.testing.assert_allclose(t_m, m, rtol=1e-7, atol=1e-7)
