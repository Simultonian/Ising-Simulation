import numpy as np
from ising.hamiltonian import Hamiltonian, parametrized_ising
from ising.lindbladian.unraveled import transpose, lowering_all_sites, LOWERING, lindbladian_operator
from qiskit.quantum_info import Pauli, SparsePauliOp


def matrix_exp_no_i(eig_vec, eig_val, eig_vec_inv, time: float):
    return (
        eig_vec
        @ np.diag(np.exp(time * eig_val))
        @ eig_vec_inv
    )

def matrix_exp(eig_vec, eig_val, eig_vec_inv, time: float):
    return (
        eig_vec
        @ np.diag(np.exp(complex(0, -1) * time * eig_val))
        @ eig_vec_inv
    )

def test_simple_H_equivalence():
    time = 1
    ham = Hamiltonian(SparsePauliOp(["Z"], coeffs=[1.0]))
    ham_mat = ham.matrix

    psi = ham.ground_state
    rho = np.outer(psi, psi.conj())
    rho_vec = rho.reshape(-1, 1)

    # l_op = -1j * np.kron(np.eye(2), ham_mat) + 1j * np.kron(ham_mat.T, np.eye(2))
    l_op = (-1j * np.kron(ham_mat, np.eye(2))) + (1j * np.kron(np.eye(2), ham_mat.T))

    eig_val, eig_vec = np.linalg.eig(l_op)
    eig_vec_inv = np.linalg.inv(eig_vec)

    op_time_matrix = matrix_exp_no_i(eig_vec, eig_val, eig_vec_inv, time)

    rho_final_vec = op_time_matrix @ rho_vec
    rho_reshaped = rho_final_vec.reshape(2, 2)


    # Hamiltonian

    h_eig_val, h_eig_vec = np.linalg.eig(ham_mat)
    h_eig_vec_inv = np.linalg.inv(h_eig_vec)

    h_time_matrix = matrix_exp(h_eig_vec, h_eig_val, h_eig_vec_inv, time)

    final_psi =  h_time_matrix @ psi
    rho_final = np.outer(final_psi, final_psi.conj())

    assert np.array_equal(rho_reshaped, rho_final)
