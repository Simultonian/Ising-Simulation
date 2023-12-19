import numpy as np

from ising.simulation.taylor.taylor import vectorized_probs, paulis_product
from qiskit.quantum_info import Pauli
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

def test_vectorized_paulis():
    paulis = [Pauli("IX"), Pauli("XI")]
    mult_inds = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    result = paulis_product(paulis, mult_inds, 1)

    expected = []
    for inds in mult_inds:
        pauli = Pauli("II")
        for ind in inds:
            pauli = pauli @ paulis[ind]
        expected.append(pauli)

    assert np.all(result == expected)

