import numpy as np
from qiskit.quantum_info import Pauli
from qiskit.circuit import Parameter
from ising.simulation.trotter.taylor import (
        get_small_k_probs, get_cap_k, 
        normalize_ham_list, get_final_term_from_sample, 
        calculate_exp_pauli, Taylor, sum_decomposition, sample_decomposition_sum)
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

def exp_ham(time, pauli):
    mat = pauli.to_matrix()
    eig_val, eig_vec = np.linalg.eig(mat)
    eig_inv = eig_vec.conj().T

    return eig_vec @ np.diag(np.exp(-1j * time * eig_val)) @ eig_inv

def check_allclose(mat, mat_list):
    return any([np.allclose(mat, matx) for matx in mat_list])

def rnd(mat):
    return np.round(mat, 2)


def test_taylor_sum_anlyt_zz():
    h_para = Parameter("h")
    error = 0.1
    delta = 0.1
    sample_count = 1000

    parametrized_ham = parametrized_ising(2, 0, 1, False)

    taylor = Taylor(parametrized_ham, h_para, error, delta)

    zz = Pauli("ZZ")
    
    h_value = 0
    rl = [1, 2, 4, 10]
    kl = [1, 2, 5, 7]
    k_max = kl[-1]
    time = 1.0 # Same as t_bar

    taylor.subsitute_h(h_value)
    taylor.construct_parametrized_circuit()

    zz_exact = exp_ham(time, zz)
    exact = taylor.get_exact_unitary(time)
    np.testing.assert_allclose(exact, zz_exact)

    decomp = sum_decomposition(taylor.paulis, time, rl[-1], taylor.coeffs, kl[-1])



    def sample_sum(r, count=sample_count):
        alphas = taylor.get_alphas(time, r, k_max)
        final = taylor.sample_v(time, r, k_max)

        for _ in range(count - 1):
            final += taylor.sample_v(time, r, k_max)

        final *= (np.sum(np.abs(alphas) ** r).real / sample_count)
        return final

    for k in kl:
        for r in rl:
            sample = sample_decomposition_sum(taylor.paulis, time, r, taylor.coeffs, k, sample_count)
            ans = np.max(np.abs(sample - exact))
            print(k, r, ans)
            # np.testing.assert_allclose(sample, exact)

    def prnt():
        print(rnd(exact))
        print(rnd(sample))
        print(rnd(decomp))

    np.testing.assert_allclose(sample, exact)
    np.testing.assert_allclose(exact, decomp)
