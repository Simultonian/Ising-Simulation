from typing import Optional
import numpy as np
from numpy.typing import NDArray

from qiskit.quantum_info import Pauli
from qiskit.circuit import Parameter

from ising.hamiltonian import Hamiltonian
from ising.hamiltonian.hamiltonian import substitute_parameter
from itertools import product as cartesian_product

import numpy as np
import numpy.testing as npt
from collections import Counter

from qiskit.quantum_info import SparsePauliOp

from ising.hamiltonian import Hamiltonian
from ising.utils import MAXSIZE, control_version
from ising.utils.constants import ONEONE, ZEROZERO
from ising.simulation.taylor.utils import (
    get_cap_k,
    calculate_exp,
    get_alphas,
    calculate_mu,
)


def calculate_decomposition_term(prod_inds, rotation_ind, paulis, t_bar, k):
    exp_pair = paulis[rotation_ind]

    cur_prob = 1.0
    cur_pauli = Pauli("I" * len(exp_pair[0]))
    for ind in prod_inds:
        pauli, prob = paulis[ind]
        assert prob > 0
        cur_prob *= prob
        cur_pauli = cur_pauli @ pauli

    exp_pauli, exp_prob = exp_pair
    assert exp_prob > 0

    exp_pauli = exp_pauli.to_matrix()
    rotated = calculate_exp(t_bar, exp_pauli, k)

    cur_prob *= exp_prob
    cur_pauli = cur_pauli.to_matrix()
    cur_pauli = cur_pauli @ rotated

    assert cur_pauli is not None
    assert cur_prob > 0

    return (cur_pauli, cur_prob)


def sum_decomposition_k_fold(paulis, t_bar, r, coeffs, cap_k):
    pairs = list(zip(paulis, coeffs))
    inds = np.arange(len(pairs))
    kth_paulis = []
    kth_probs = []
    k_probs = []

    alphas = np.array(get_alphas(t_bar, cap_k, r))
    t_bar = t_bar / r

    for k in range(0, cap_k + 1):
        if k % 2 == 1:
            # Odd k can not be sampled.
            k_probs.append(0)
            kth_probs.append([])
            kth_paulis.append([])
            continue

        alpha_term = alphas[k]

        if t_bar == 0.0:
            if k > 0:
                assert alpha_term == 0.0

        mult_inds = cartesian_product(inds, repeat=k + 1)
        terms = []
        probs = []

        for term_inds in mult_inds:
            prod_inds, rotation_ind = term_inds[:-1], term_inds[-1]

            cur_pauli, cur_prob = calculate_decomposition_term(
                prod_inds, rotation_ind, pairs, t_bar, k
            )

            if alpha_term < 0:
                # Transferring the alpha negative to the term.
                cur_pauli *= -1

            assert isinstance(cur_pauli, np.ndarray)
            terms.append(cur_pauli)
            probs.append(cur_prob)

        probs = np.array(probs).real
        npt.assert_allclose(np.sum(probs), 1, atol=1e-5, rtol=1e-5)

        k_probs.append(abs(alpha_term))

        kth_probs.append(probs)
        kth_paulis.append(terms)

    k_probs = np.array(k_probs)
    k_probs /= np.sum(k_probs)

    return (kth_paulis, kth_probs, k_probs)


def taylor_observation(
    paulis: list[Pauli],
    coeffs: NDArray,
    time: float,
    error: float,
    obs,
    psi_init,
    success: float,
):
    beta = np.sum(np.array(coeffs))
    coeffs /= beta

    t_bar = time * beta
    # For t_bar < 1, r is too small to get accurate results.
    r = max(20, int(5 * np.ceil(t_bar) ** 2))

    # TODO obs_norm
    obs_norm = 1
    cap_k = get_cap_k(t_bar, obs_norm=obs_norm, eps=error)

    print(f"Computing decomposition for t_bar={t_bar} t={time} r={r} k={cap_k}")
    kth_paulis, kth_probs, k_probs = sum_decomposition_k_fold(
        paulis, t_bar, r, coeffs, cap_k
    )
    print("Decomposition complete")

    def get_k_terms(k):
        return np.random.choice(len(kth_probs[k]), p=kth_probs[k], size=r)

    def get_unitary(k, ind: int):
        return kth_paulis[k][ind]

    def control_unitary(k, ind: int, control_val: int):
        """
        Calculates |0><0| U + |1><1| I if control_val = 0
        Calculates |0><0| I + |1><1| U if control_val = 1
        """
        return control_version(get_unitary(k, ind), control_val)

    def post_v1(k):
        final_psi = psi_init.copy()
        inds = get_k_terms(k)

        for ind in inds:
            v1 = control_unitary(k, ind, control_val=1)
            final_psi = v1 @ final_psi

        npt.assert_almost_equal(np.sum(np.abs(final_psi) ** 2), 1)
        return final_psi

    def post_v1v2(k1: int, k2: int):
        final_psi = post_v1(k1)
        i2s = get_k_terms(k2)

        for i2 in i2s:
            v2 = control_unitary(k2, i2, control_val=0)
            final_psi = v2 @ final_psi

        npt.assert_almost_equal(np.sum(np.abs(final_psi) ** 2), 1)
        return final_psi

    total_count = 0
    count = int(
        8 * np.ceil(((obs_norm**2) * (np.log(2 / (1 - success)))) / (error**2))
    )
    results = []

    sample_ks = Counter(
        [tuple(x) for x in np.random.choice(cap_k + 1, p=k_probs, size=(count, 2))]
    )

    print(f"Time:{time} Iterations:{count}")

    for k1, k2 in sorted(sample_ks.keys()):
        k_count = sample_ks[(k1, k2)]
        if total_count % 100 == 0:
            print(f"running: {total_count} out of {count}")
        total_count += k_count

        for _ in range(k_count):
            final_psi = post_v1v2(k1, k2)
            final_rho = np.outer(final_psi, final_psi.conj())

            result = np.trace(np.abs(obs @ final_rho))
            results.append(result)

    assert total_count == count

    magn_h = calculate_mu(results, count, [1])
    return magn_h


class Taylor:
    """
    Creates the entire decomposition and then samples from that.
    """

    def __init__(self, ham: Hamiltonian, h_para: Parameter, error: float, **kwargs):
        self._ham = ham
        self.h_para = h_para
        self.ham: Optional[Hamiltonian] = None

        self.num_qubits = ham.sparse_repr.num_qubits
        self.error = error

        self.success = kwargs.get("success", 0.9)

    @property
    def ground_state(self) -> NDArray:
        if self.ham is None:
            raise ValueError(
                "h value has not been substituted, qiskit does not support parametrized Hamiltonians."
            )
        return self.ham.ground_state

    def subsitute_h(self, h_val: float) -> None:
        self.ham = substitute_parameter(self._ham, self.h_para, h_val)
        paulis, coeffs = [], []

        for pauli, _coeff in zip(self.ham.paulis, self.ham.coeffs):
            assert _coeff.imag == 0
            coeff = _coeff.real
            if coeff < 0:
                paulis.append(-pauli)
                coeffs.append(-coeff)
            elif coeff > 0:
                paulis.append(pauli)
                coeffs.append(coeff)

        self.paulis, self.coeffs = paulis, np.array(coeffs)

    def substitute_obs(self, obs: Hamiltonian):
        obs_x = SparsePauliOp(["X"], np.array([1.0]))
        run_obs = obs_x.tensor(obs.sparse_repr)
        self.obs = run_obs.to_matrix()

    def get_observation(self, time, psi_init):
        if self.ham is None:
            raise ValueError("Parameter not substituted.")
        if self.obs is None:
            raise ValueError("Observable not substituted.")

        return taylor_observation(
            self.paulis,
            self.coeffs,
            time,
            self.error,
            self.obs,
            psi_init,
            self.success,
        )

    def get_observations(self, psi_init: NDArray, times: list[float]):
        return [self.get_observation(time, psi_init) for time in times]
