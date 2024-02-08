from typing import Optional
from functools import lru_cache
import numpy as np
from numpy.typing import NDArray

from qiskit.quantum_info import PauliList
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
from ising.simulation.taylor.utils import (
    get_cap_k,
    calculate_exp,
    get_alphas,
    calculate_mu,
)

from tqdm import tqdm


def vectorized_probs(coeffs, mult_inds):
    """
    Calculate probability of each indices in a vectorized manner using numpy.
    """
    return np.prod(coeffs[mult_inds], axis=1).real


def product_mult_inds(paulis, mult_inds):
    """
    Given a 2D array, return the product of all paulis in each row
    """
    num_qubits = len(paulis[0])
    terms_count = mult_inds.shape[0]

    if terms_count == 0:
        return PauliList(["I" * num_qubits] * terms_count)
    if terms_count == 1:
        return PauliList([paulis[ind] for ind in mult_inds.T[0]])

    target = PauliList(["I" * num_qubits] * terms_count)
    for inds in mult_inds.T:
        target.compose(PauliList([paulis[ind] for ind in inds]), inplace=True)

    return target


def sum_decomposition_k_fold(paulis, coeffs, cap_k):
    cart_inds = np.arange(len(paulis))
    kth_paulis = []
    kth_exps = []
    kth_probs = []

    for k in range(0, cap_k + 1):
        if k % 2 == 1:
            # Odd k can not be sampled.
            kth_probs.append([])
            kth_exps.append([])
            kth_paulis.append([])
            continue

        mult_inds = np.array(list(cartesian_product(cart_inds, repeat=k + 1)))

        kth_probs.append(vectorized_probs(coeffs, mult_inds))
        kth_exps.append(paulis[mult_inds[:, -1]])

        pauli_terms = product_mult_inds(paulis, mult_inds[:, :-1])
        kth_paulis.append(pauli_terms)

    return (kth_paulis, kth_exps, kth_probs)


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
        """
        Setting up the Hamiltonian and the terms to sample.
        """
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

        self.paulis, self.coeffs = PauliList(paulis), np.array(coeffs)
        self.beta = np.sum(np.array(self.coeffs))
        self.coeffs /= self.beta

    def set_up_decomposition(self, max_time: float) -> None:
        t_bar = max_time * self.beta
        # TODO obs_norm
        obs_norm = 1
        self.cap_k = get_cap_k(t_bar, obs_norm=obs_norm, eps=self.error)

        print(f"Computing decomposition for t_bar={t_bar} t={max_time} k={self.cap_k}")
        self.kth_paulis, self.kth_exps, self.kth_probs = sum_decomposition_k_fold(
            self.paulis, self.coeffs, self.cap_k
        )
        print("Decomposition complete")

    def substitute_obs(self, obs: Hamiltonian):
        obs_x = SparsePauliOp(["X"], np.array([1.0]))
        run_obs = obs_x.tensor(obs.sparse_repr)
        self.obs = run_obs.to_matrix()

    def get_observation(self, time, psi_init):
        if self.ham is None:
            raise ValueError("Parameter not substituted.")
        if self.obs is None:
            raise ValueError("Observable not substituted.")

        return self.taylor_observation(
            time,
            psi_init,
        )

    def get_observations(self, psi_init: NDArray, times: list[float]):
        if self.ham is None:
            raise ValueError("Parameter not substituted.")
        if self.obs is None:
            raise ValueError("Observable not substituted.")

        return [self.taylor_observation(time, psi_init) for time in times]

    @lru_cache(maxsize=None)
    def pauli_to_matrix(self, pauli):
        return pauli.to_matrix()

    def get_k_terms(self, k, count, r):
        return Counter(
            [
                tuple(x)
                for x in np.random.choice(
                    len(self.kth_probs[k]), p=self.kth_probs[k], size=(count, r)
                )
            ]
        )

    @lru_cache(maxsize=None)
    def get_unitary(self, time: float, k: int, ind: int):
        rotated = calculate_exp(time, self.pauli_to_matrix(self.kth_exps[k][ind]), k)
        return self.pauli_to_matrix(self.kth_paulis[k][ind]) @ rotated

    @lru_cache(maxsize=None)
    def control_unitary(self, time: float, k, ind: int, control_val: int):
        """
        Calculates |0><0| U + |1><1| I if control_val = 0
        Calculates |0><0| I + |1><1| U if control_val = 1
        """
        return control_version(self.get_unitary(time, k, ind), control_val)

    @lru_cache(maxsize=MAXSIZE)
    def post_v1(self, time: float, k, inds: tuple[int, ...]):
        final_psi = self.psi_init.copy()

        for ind in inds:
            v1 = self.control_unitary(time, k, ind, control_val=1)
            final_psi = v1 @ final_psi

        npt.assert_almost_equal(np.sum(np.abs(final_psi) ** 2), 1)
        return final_psi

    def post_v1v2(
        self,
        time: float,
        ks: tuple[int, int],
        i1s: tuple[int, ...],
        i2s: tuple[int, ...],
    ):
        k1, k2 = ks
        final_psi = self.post_v1(time, k1, i1s)

        for i2 in i2s:
            v2 = self.control_unitary(time, k2, i2, control_val=0)
            final_psi = v2 @ final_psi

        npt.assert_almost_equal(np.sum(np.abs(final_psi) ** 2), 1)
        return final_psi

    def circuit_depth(self, time:float) -> int:
        t_bar = time * self.beta
        # For t_bar < 1, r is too small to get accurate results.
        return max(20, int(10 * np.ceil(t_bar) ** 2))

    def apply_lcu(self, time, final_psi, control_val):
        """
        Samples V from the LCU and applies it to `psi_init` with given
        control value. The groundstate preparation algorithm calls this
        function twice for V1, V2. It is done separately because `time` is
        independently sampled for V1 and V2 unlike LCU simulation.
        ---
        `final_psi` has already been copied in the parent function call
        """
        t_bar = time * self.beta
        # For t_bar < 1, r is too small to get accurate results.
        r = max(20, int(10 * np.ceil(t_bar) ** 2))

        evo_time = np.round(t_bar / r, 6)

        alphas = get_alphas(t_bar, self.cap_k, r)
        k_probs = np.abs(alphas)
        k_probs /= np.sum(k_probs)

        k = np.random.choice(self.cap_k + 1, p=k_probs)

        k_terms = np.random.choice(len(self.kth_probs[k]), p=self.kth_probs[k], size=r)
        for k_term in k_terms:
            v = self.control_unitary(evo_time, k, k_term, control_val)
            final_psi = v @ final_psi

        neg = 1
        if alphas[k] < 0 and r % 2 == 1:
            neg *= -1

        return neg * final_psi

    def taylor_observation(
        self,
        time: float,
        psi_init,
    ):
        self.post_v1.cache_clear()
        self.control_unitary.cache_clear()
        self.get_unitary.cache_clear()

        self.psi_init = psi_init
        t_bar = time * self.beta
        # For t_bar < 1, r is too small to get accurate results.
        r = max(20, int(5 * np.ceil(t_bar) ** 2))

        # TODO obs_norm
        obs_norm = 1

        alphas = get_alphas(t_bar, self.cap_k, r)
        k_probs = np.abs(alphas)
        k_probs /= np.sum(k_probs)

        count = int(
            8
            * np.ceil(
                ((obs_norm**2) * (np.log(2 / (1 - self.success)))) / (self.error**2)
            )
        )
        results = []

        sample_ks = Counter(
            [
                tuple(x)
                for x in np.random.choice(self.cap_k + 1, p=k_probs, size=(count, 2))
            ]
        )

        print(f"Time:{time} Iterations:{count}")

        evo_time = np.round(t_bar / r, 6)

        with tqdm(total=count) as pbar:
            for k1, k2 in sorted(sample_ks.keys()):
                k_count = sample_ks[(k1, k2)]

                k1_terms = self.get_k_terms(k1, k_count, r)
                k2_terms = self.get_k_terms(k2, k_count, r)

                for k1_term, k2_term in zip(k1_terms, k2_terms):
                    final_psi = self.post_v1v2(evo_time, (k1, k2), k1_term, k2_term)

                    neg = 1
                    if alphas[k1] < 0 and len(k1_term) % 2 == 1:
                        neg *= -1

                    if alphas[k2] < 0 and len(k2_term) % 2 == 1:
                        neg *= -1

                    final_psi = neg * final_psi

                    final_rho = np.outer(final_psi, final_psi.conj())

                    result = np.trace(np.abs(self.obs @ final_rho))
                    pbar.update(1)
                    results.append(result)

        magn_h = calculate_mu(results, count, [1])
        return magn_h
