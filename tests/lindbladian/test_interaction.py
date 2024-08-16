import numpy as np
from qiskit.quantum_info import SparsePauliOp
from ising.lindbladian.simulation.utils import save_interaction_hams, load_interaction_hams, interaction_hamiltonian, interaction_hamiltonian_sparse



def test_2_qubit_save_load():
    qubits = 3
    gamma = 0.1
    save_interaction_hams(qubits)

    results = load_interaction_hams(qubits, gamma)
    actual_mats = interaction_hamiltonian(qubits, gamma)

    for res, actual_mat in zip(results, actual_mats):
        res_mat = res.to_matrix()

        assert np.allclose(actual_mat, res_mat)


SIGMA_MINUS = np.array([[0, 1], [0, 0]])
SIGMA_PLUS = np.array([[0, 0], [1, 0]])

SIGMA_PLUS_SPARSE = SparsePauliOp(["X", "Y"], [0.5, -0.5j])
SIGMA_MINUS_SPARSE = SparsePauliOp(["X", "Y"], [0.5, 0.5j])
EYE = SparsePauliOp(["I"], [1])


def test_sigmas():
    assert np.allclose(SIGMA_MINUS, SIGMA_MINUS_SPARSE.to_matrix())
    assert np.allclose(SIGMA_PLUS, SIGMA_PLUS_SPARSE.to_matrix())
