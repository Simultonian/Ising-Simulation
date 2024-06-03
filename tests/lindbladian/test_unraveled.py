import numpy as np
from ising.hamiltonian import Hamiltonian
from ising.lindbladian.unraveled import transpose, lowering_all_sites, LOWERING
from qiskit.quantum_info import Pauli, SparsePauliOp

def test_tensor():
    ham = SparsePauliOp(["IX", "XI", "ZZ"], coeffs=[1, 2, 3])
    res = -1j * SparsePauliOp("I").tensor(ham)
    expected = SparsePauliOp(["IIX", "IXI", "IZZ"], coeffs=[-1j, -2j, -3j])
    assert res == expected

def test_transpose():

    ham = SparsePauliOp(["XI", "ZZ", "YI", "YY"], coeffs=[1, 1, 1, 1])
    
    res = transpose(ham)

    expected = SparsePauliOp(["XI", "ZZ", "YI", "YY"], coeffs=[1, 1, -1, 1])
    assert res == expected

def test_tensor_two():
    # Define matrices A and B
    A = np.array([[1, 0], [0, 1]])
    B = np.array([[0, 5], [6, 7]])

    expected = np.array(
            [[0, 5, 0, 0],
             [6, 7, 0, 0],
             [0, 0, 0, 5],
             [0, 0, 6, 7],
                ])

    # Compute the tensor product of A and B
    result = np.kron(A, B)
    assert np.array_equal(expected, result)

def test_lowering_ops_one():
    res = lowering_all_sites(1)

    assert len(res) == 1
    assert np.array_equal(res[0], LOWERING)

def test_lowering_ops_two():
    res = lowering_all_sites(2)

    assert len(res) == 2
    assert np.array_equal(res[0], np.kron(LOWERING, np.eye(2)))
    assert np.array_equal(res[1], np.kron(np.eye(2), LOWERING))

def test_lowering_ops_three():
    res = lowering_all_sites(3)

    assert len(res) == 3
    assert np.array_equal(res[0], np.kron(LOWERING, np.eye(4)))
    assert np.array_equal(res[1], np.kron(np.kron(np.eye(2), LOWERING), np.eye(2)))
    assert np.array_equal(res[2], np.kron(np.eye(4), LOWERING))
