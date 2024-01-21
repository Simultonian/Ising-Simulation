import numpy as np

from ising.simulation.taylor.taylor import vectorized_probs, product_mult_inds
from qiskit.quantum_info import Pauli, PauliList
from itertools import product as cartesian_product


def test_cartesian():
    coeffs = [0, 1]
    inds = np.arange(len(coeffs))
    mult_inds = np.array(list(cartesian_product(inds, repeat=2)))

    expected = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    assert np.all(expected == mult_inds)


def test_vectorized_probs():
    coeffs = np.array([1, 2, 3, 4, 5])
    mult_inds = np.array([[0, 1], [1, 2], [3, 4], [1, 4]])

    result = vectorized_probs(coeffs, mult_inds)

    expected = []
    for inds in mult_inds:
        ans = 1
        for ind in inds:
            ans *= coeffs[ind]
        expected.append(ans)

    expected = np.array(expected)
    np.all(result == expected)


def test_pauli_list_product_simple():
    paulis = PauliList([Pauli("IX"), Pauli("XI")])
    mult_inds = np.array([[0], [1]])

    result = product_mult_inds(paulis, mult_inds)
    target = PauliList([Pauli("IX"), Pauli("XI")])

    assert np.all(result == target)


def test_pauli_list_product():
    paulis = PauliList([Pauli("IX"), Pauli("XI")])
    mult_inds = np.array([[0, 1], [1, 1]])

    result = product_mult_inds(paulis, mult_inds)
    target = PauliList([Pauli("XX"), Pauli("II")])

    assert np.all(result == target)


def test_pauli_list_product_unequal():
    paulis = PauliList([Pauli("IX"), Pauli("XI")])
    mult_inds = np.array([[0, 1, 1], [1, 1, 0]])

    result = product_mult_inds(paulis, mult_inds)
    target = PauliList([Pauli("IX"), Pauli("IX")])

    assert np.all(result == target)


def test_pauli_list_negative():
    paulis = PauliList([Pauli("-IX"), Pauli("XI")])
    result = -paulis

    target = PauliList([Pauli("IX"), Pauli("-XI")])

    assert np.all(result == target)
