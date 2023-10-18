import numpy as np
from qiskit.quantum_info import Pauli
from qiskit.circuit import Parameter
from ising.simulation.trotter.taylor import (
        get_small_k_probs, get_cap_k, 
        normalize_ham_list, get_final_term_from_sample, 
        calculate_exp_pauli, Taylor, sum_decomposition)
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



def exp_ham(time, pauli):
    mat = pauli.to_matrix()
    eig_val, eig_vec = np.linalg.eig(mat)
    eig_inv = eig_vec.conj().T

    return eig_vec @ np.diag(np.exp(-1j * time * eig_val)) @ eig_inv



def check_allclose(mat, mat_list):
    return any([np.allclose(mat, matx) for matx in mat_list])



def rnd(mat):
    return np.round(mat, 2)


def test_taylor_sum_convergence():
    h_para = Parameter("h")
    return
    error = 0.1
    delta = 0.1
    sample_count = 10000

    parametrized_ham = parametrized_ising(2, h_para, 0, False)

    taylor = Taylor(parametrized_ham, h_para, error, delta)

    xi = Pauli("XI")
    ix = Pauli("IX")
    xx = Pauli("XX")
    xx_mat = xx.to_matrix()
    
    time1 = np.arccos(1 / np.sqrt(2))
    
    k01 = exp_ham(time1, ix)
    k02 = exp_ham(time1, xi)
    k0 = [k01, k02]

    time2 = np.arccos(3 / np.sqrt(10))

    k2 = []
    k2.append(exp_ham(time2, ix))
    k2.append(exp_ham(time2, xi))
    k2.append(xx_mat @ exp_ham(time2, ix))
    k2.append(xx_mat @ exp_ham(time2, xi))

    k2 = [-x for x in k2]


    all_ks = k0 + k2



    for h_value in [1.0, 0.9, 2.0]:
        taylor.subsitute_h(h_value)
        taylor.construct_parametrized_circuit()

        for time in [1.0/4, 10.0, 5.0]:
            alphas = taylor.get_alphas(time)

            alyt_0 = (alphas[0] / len(k0)) * np.sum(k0, axis=0)
            alyt_2 = (-alphas[2] / len(k2)) * np.sum(k2, axis=0)
            finalyt = alyt_0 + alyt_2
 
            expected = taylor.get_exact_unitary(time)
            exact = exp_ham(1/4, ix) @ exp_ham(1/4, xi)
            np.testing.assert_almost_equal(expected, exact)

            # MATCHING
            # np.testing.assert_almost_equal(expected, finalyt)

            final = None
            for _ in range(sample_count):
                res = taylor.sample_v(time)
                # assert check_allclose(res, all_ks)
                
                if final is None:
                    final = res
                else:
                    final += res

            # final /= global_phase(final)
            final *= (np.sum(np.abs(alphas)).real/sample_count)
            np.testing.assert_almost_equal(finalyt, final)
            np.testing.assert_almost_equal(expected, final)



def test_taylor_sum_anlyt_xx():
    return
    h_para = Parameter("h")
    error = 0.1
    delta = 0.1
    sample_count = 10000

    parametrized_ham = parametrized_ising(2, h_para, 0, False)

    taylor = Taylor(parametrized_ham, h_para, error, delta)

    xi = Pauli("XI")
    ix = Pauli("IX")
    xx = Pauli("XX")
    xx_mat = xx.to_matrix()
    
    time1 = np.arccos(1 / np.sqrt(2))
    
    k01 = exp_ham(time1, ix)
    k02 = exp_ham(time1, xi)
    k0 = [k01, k02]

    time2 = np.arccos(3 / np.sqrt(10))

    k2 = []
    k2.append(exp_ham(time2, ix))
    k2.append(exp_ham(time2, xi))
    k2.append(xx_mat @ exp_ham(time2, ix))
    k2.append(xx_mat @ exp_ham(time2, xi))

    k2 = [-x for x in k2]


    all_ks = k0 + k2

    r = 10

    for h_value in [1.0, 0.9, 2.0]:
        beta = h_value * 2
        taylor.subsitute_h(h_value)
        taylor.construct_parametrized_circuit()

        for time in [0.5, 1.0, 10.0, 5.0]:
            exact = taylor.get_exact_unitary(time)

            decomp = sum_decomposition(taylor.paulis, time, r, beta, taylor.coeffs, 6)

            def sample_sum(r, count=sample_count):
                alphas = taylor.get_alphas(time, r)
                final = None
                for _ in range(count):
                    res = taylor.sample_v(time, r)
                    # assert check_allclose(res, all_ks)
                    
                    if final is None:
                        final = res
                    else:
                        final += res

                final *= (np.sum(np.abs(alphas) ** r).real/sample_count)
                return final

            final = sample_sum(r)
            def prnt():
                print(rnd(exact))
                print(rnd(final))
                print(rnd(decomp))

            np.testing.assert_allclose(exact, decomp)


def test_taylor_sum_anlyt():
    h_para = Parameter("h")
    error = 0.1
    delta = 0.1
    sample_count = 10000

    parametrized_ham = parametrized_ising(2, h_para, -1, False)

    taylor = Taylor(parametrized_ham, h_para, error, delta)

    for h_value in [1.0, 0.9, 2.0]:
        taylor.subsitute_h(h_value)
        taylor.construct_parametrized_circuit()
        beta = np.sum(np.abs(taylor.ham_subbed.coeffs))

        r = 1
        exacts, finals, decomps = [], [], []
        for time in [3.0]:
            exact = taylor.get_exact_unitary(time)
            
            t_bar = beta * time
            decomp = sum_decomposition(taylor.paulis, t_bar, r, taylor.coeffs, 6)

            def sample_sum(r=None, count=sample_count):
                alphas = taylor.get_alphas(time, r)
                final = None
                for _ in range(count):
                    res = taylor.sample_v(time, r)
                    # assert check_allclose(res, all_ks)
                    
                    if final is None:
                        final = res
                    else:
                        final += res

                if r is None:
                    r = int(np.ceil(t_bar ** 2))

                final *= (np.sum(np.abs(alphas) ** r).real/sample_count)
                return final

            # final = sample_sum(r)
            # def prnt_in():
            #     print(rnd(exact))
            #     print(rnd(final))
            #     print(rnd(decomp))

            def loop_decomp(x, y, s):
                for a in range(x, y, s):
                    print(rnd(sum_decomposition(taylor.paulis, t_bar, a, taylor.coeffs, 6)))

            def loop_sample(x, y, s):
                for a in range(x, y, s):
                    print(rnd(sample_sum(a,1000)))

            assert False
            exacts.append(exact)
            finals.append(final)
            decomps.append(decomp)

        def prnt(i):
            print(rnd(exacts[i]))
            print(rnd(finals[i]))
            print(rnd(decomps[i]))

        assert False
