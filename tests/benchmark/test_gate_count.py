import numpy as np

from qiskit.quantum_info import SparsePauliOp
from ising.hamiltonian import Hamiltonian

from ising.benchmark.gate_err import (
    trotter_gate_count,
    qdrift_gate_count,
    taylor_gate_count,
)


def test_trotter():
    sparse_ham = SparsePauliOp(["Z", "X"], coeffs=np.array([1.0, -2.0]))
    ham = Hamiltonian(sparse_ham, False)
    time = 1
    err = 0.1

    result = trotter_gate_count(ham, time, err)

    answer = ((2**3) * (2 * 1) ** 2) * 10

    assert result == answer


def test_qdrift():
    sparse_ham = SparsePauliOp(["Z", "X"], coeffs=np.array([1.0, -2.0]))
    ham = Hamiltonian(sparse_ham, False)

    time = 1
    err = 0.1

    result = qdrift_gate_count(ham, time, err)

    answer = ((3 * 1) ** 2) * 10

    assert result == answer


def test_taylor():
    sparse_ham = SparsePauliOp(["Z", "X"], coeffs=np.array([1.0, -2.0]))
    ham = Hamiltonian(sparse_ham, False)

    time = 1
    err = 0.1
    obs_norm = 1

    result = taylor_gate_count(ham, time, err, obs_norm)

    answer = np.ceil((3 * 1) ** 2 * np.log(3 * 10) / np.log(np.log(3 * 10)))

    assert result == answer
