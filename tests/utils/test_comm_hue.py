import numpy as np
from qiskit.quantum_info import SparsePauliOp, Pauli
from ising.utils.commutator.commutator_hueristic import r_first_order as hue_first_order
from ising.utils.commutator.commutator_brute import (
    commutator_r_first_order as brute_first_order,
    alpha_commutator_first_order,
)
from ising.hamiltonian.ising_one import parametrized_ising


def test_match_brute_hue():
    num_qubits, h = 7, 0.125
    time, error = 1, 0.1
    hamiltonian = parametrized_ising(num_qubits, h)

    sorted_pairs = list(
        sorted(
            zip(hamiltonian.paulis, hamiltonian.coeffs),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
    )
    sorted_pairs = [(x, y.real) for (x, y) in sorted_pairs]

    brute = brute_first_order(hamiltonian.sparse_repr, time, error)
    hue = hue_first_order(sorted_pairs, time=time, error=0.1)

    assert brute == hue


def test_match_brute_hue_delta():
    num_qubits, h = 7, 0.125
    time, error = 1, 0.1
    delta = 0.01

    hamiltonian = parametrized_ising(num_qubits, h)

    sorted_pairs = list(
        sorted(
            zip(hamiltonian.paulis, hamiltonian.coeffs),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
    )
    sorted_pairs = [(x, y.real) for (x, y) in sorted_pairs]

    brute = brute_first_order(hamiltonian.sparse_repr, time, error)
    hue = hue_first_order(sorted_pairs, time=time, error=0.1, delta=delta)

    assert abs(brute - hue) < delta


def test_match_brute_hue_delta_cutoff():
    num_qubits, h = 7, 0.125
    time, error = 1, 0.1
    delta = 0.01

    hamiltonian = parametrized_ising(num_qubits, h)

    sorted_pairs = list(
        sorted(
            zip(hamiltonian.paulis, hamiltonian.coeffs),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
    )
    sorted_pairs = [(x, y.real) for (x, y) in sorted_pairs]
    l = len(sorted_pairs)

    alpha_comm = alpha_commutator_first_order(hamiltonian.sparse_repr)
    brute = brute_first_order(hamiltonian.sparse_repr, time, error)
    hue = hue_first_order(
        sorted_pairs, time=time, error=0.1, delta=delta, cutoff_count=25 * l
    )

    brute_error = alpha_comm * (time**2) / brute
    hue_error = alpha_comm * (time**2) / hue

    assert abs(brute_error - hue_error) < delta
