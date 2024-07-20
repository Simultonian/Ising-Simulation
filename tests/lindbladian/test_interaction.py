import numpy as np
from ising.lindbladian.simulation.utils import save_interaction_hams, load_interaction_hams, interaction_hamiltonian



def test_2_qubit_save_load():
    qubits = 2
    save_interaction_hams(qubits)
    results = load_interaction_hams(qubits)
    actual_mats = interaction_hamiltonian(qubits)

    for res, actual_mat in zip(results, actual_mats):
        res_mat = res.to_matrix()

        assert np.allclose(actual_mat, res_mat)
