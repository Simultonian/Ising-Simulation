import numpy as np
from qiskit.quantum_info import Pauli
from qiskit.circuit import Parameter
from ising.simulation.trotter.taylor import (
        get_small_k_probs, get_cap_k, 
        normalize_ham_list, get_final_term_from_sample, 
        calculate_exp_pauli, Taylor)
from ising.hamiltonian import parametrized_ising



def test_k_probs():
    k_vals = get_small_k_probs(t_bar=1,r=1,cap_k=2)

    expected = np.array([ 1.41421356+0.j,  0.        +0.j, -0.52704628+0.j])
    # expected /= np.sum(np.abs(expected))
    assert np.allclose(k_vals, expected)


def test_k_sum():
    time = 10
    beta = 4
    obs_norm = 1
    error = 0.1

    t_bar = time * beta
    r = t_bar ** 2
    cap_k = get_cap_k(t_bar, obs_norm, error)
    alphas = np.abs(get_small_k_probs(t_bar=t_bar, r=r, cap_k=cap_k))

    k_probs = np.abs(alphas)
    k_probs /= np.sum(k_probs)

    assert np.allclose(np.sum(k_probs), 1.0)


def test_normalize_ham_list():
    pauli_map = {Pauli("X"): -1, Pauli("Y"): 0j, Pauli("Z"): 0.5}
    paulis, coeffs, beta = normalize_ham_list(pauli_map)

    exp_paulis = [Pauli("-X"), Pauli("Y"), Pauli("Z")]
    exp_coeffs = [2/3, 0, 1/3]

    assert paulis == exp_paulis
    assert np.allclose(coeffs, exp_coeffs)
    assert beta == 1.5


def test_sample_ham_list():
    pauli_map = {Pauli("X"): -1, Pauli("Y"): 0j, Pauli("Z"): 0.5}
    sample_size = 1000
    paulis, coeffs, beta = normalize_ham_list(pauli_map)
    pauli_inds = np.arange(len(paulis))

    sampled_paulis = np.random.choice(pauli_inds, p=coeffs, size=sample_size)
    assert len(sampled_paulis) == sample_size
    for x in sampled_paulis:
        assert x in pauli_inds

def test_exp_pauli_id():
    t_bar = 1
    r = 1
    k = 0
    pauli = Pauli("I").to_matrix()
    result = calculate_exp_pauli(t_bar, r, k, pauli)

    expected = np.identity(2) * (1 - 1j) / np.sqrt(2)

    assert np.allclose(result, expected)

def test_get_final_term_from_sample():
    indices = [0, 1]
    rotation_ind = 0
    paulis = [Pauli("X"), Pauli("Z")]
    coeffs = [0.5, 0.5]
    alpha = 1
    t_bar = 1
    r = 1
    k = 2

    x = get_final_term_from_sample(indices, rotation_ind, paulis, coeffs, alpha, t_bar, r, k)

import cmath
def global_phase(a: np.ndarray) -> complex:
    # get the phase of the first non-zero value in the matrix
    for row in a:
        for x in row:
            if np.abs(x) > 1e-3:
                theta = cmath.phase(x)
                # phase is e^{i\theta}
                phase = np.exp(0 + 1j * theta)
                # print(x, theta, phase)
                return phase

    return 1 + 0j


def test_taylor_sum_convergence():
    h_para = Parameter("h")
    error = 0.1
    delta = 0.1
    sample_count = 10000

    parametrized_ham = parametrized_ising(2, h_para)

    taylor = Taylor(parametrized_ham, h_para, error, delta)



    zz = Pauli("ZZ")
    zz_mat = zz.to_matrix()
    e_val, e_vec = np.linalg.eig(zz_mat)
    # e_inv = np.linalg.inv(e_vec)
    e_inv = e_vec.conj().T


    def zz_exp(time):
        return e_vec @ np.diag(np.exp(complex(0, 1) * time * e_val)) @ e_inv 

    def rzz(p):
        power = 1j * np.arccos(1/p)
        return zz_exp(power)

    
    p1, p2 = np.sqrt(2), np.sqrt(10) / 6
    term1, term2 = rzz(p1), -rzz(p2*2)

    sum_term = p1 * term1 + p2 * term2
    expected = zz_exp(1.0)

    # assert np.allclose(sum_term, expected)


    for h_value in [0.0, 1.0, 0.9, 2.0]:
        taylor.subsitute_h(h_value)
        taylor.construct_parametrized_circuit()

        for time in [1.0, 10.0, 5.0]:
            alphas = taylor.get_alphas(time)

            final = None
            for _ in range(sample_count):
                res = taylor.sample_v(time)
                if final is None:
                    final = res
                else:
                    final += res

            # final /= global_phase(final)
            final *= (np.sum(np.abs(alphas)).real/sample_count)

 
            expected = zz_exp(time)
            np.testing.assert_almost_equal(expected, final)

