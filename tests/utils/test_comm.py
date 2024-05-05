import numpy as np
from qiskit.quantum_info import SparsePauliOp, Pauli

def test_mult():
    x, y = Pauli("X"), Pauli("Y")
    ca, cb = 1, 2
    ham = SparsePauliOp([x, y], [1.0, 2.0])

    ta = (ca * cb), (x @ y)
    tb = (-cb * ca), (y @ x)

    print(ta, tb)
    print(SparsePauliOp([ta[1], tb[1]], [ta[0], tb[0]]))


def test_sort():
    paulis = [Pauli("XI"), Pauli("IX"), Pauli("XY"), Pauli("ZZ")]
    coeffs = [-1, 2, 3, 0]

    sorted_pairs = list(sorted(zip(paulis, coeffs), key=lambda x: abs(x[1])))
    
    first = sorted_pairs[0]
    assert first[0] == Pauli("ZZ")
    assert first[1] == 0

    second = sorted_pairs[1]
    assert second[0] == Pauli("XI")
    assert second[1] == -1


def _pauli_commute(a: Pauli, b: Pauli):
    x1, z1 = a._x, a._z

    a_dot_b = np.mod((x1 & b._z).sum(axis=1), 2)
    b_dot_a = np.mod((b._x & z1).sum(axis=1), 2)

    return a_dot_b == b_dot_a

def _pauli_multi_commute(a: Pauli, xs, zs):
    x1, z1 = a._x, a._z

    a_dot_b = np.mod((x1 & zs).sum(axis=1), 2)
    b_dot_a = np.mod((xs & z1).sum(axis=1), 2)

    return np.all(a_dot_b == b_dot_a, axis=1)

def test_multi_commute():
    paulis = [Pauli("XI"), Pauli("IX"), Pauli("ZI")]
    pauli = Pauli("XX")

    all_x = np.array([p._x for p in paulis])
    all_y = np.array([p._z for p in paulis])

    assert _pauli_commute(paulis[0], pauli)
    assert _pauli_commute(paulis[1], pauli)
    assert not _pauli_commute(paulis[2], pauli)

    ret = _pauli_multi_commute(pauli, all_x, all_y)
    assert all(ret == [True, True, False])
