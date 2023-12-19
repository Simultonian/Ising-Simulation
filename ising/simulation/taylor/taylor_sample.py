from typing import Optional
import numpy as np
from numpy.typing import NDArray

from qiskit.quantum_info import Pauli
from qiskit.circuit import Parameter

from ising.hamiltonian import Hamiltonian
from ising.hamiltonian.hamiltonian import substitute_parameter

import numpy as np
import numpy.testing as npt
from collections import Counter

from qiskit.quantum_info import SparsePauliOp

from ising.hamiltonian import Hamiltonian
from ising.utils import control_version
from ising.simulation.taylor.utils import (
    get_cap_k,
    calculate_exp,
    get_alphas,
    calculate_mu,
)

from tqdm import tqdm


def sample_for_fixed_k(paulis, coeffs, t_bar, r, k, alpha):
    """
    For a fixed `k` a unitary from LCU is sampled. `alpha` is required for the
    sake of changing the sign of the Pauli matrix.
    """
    t_bar = t_bar / r

    # Sampling the exponential term.
    exp_ind = np.random.choice(len(coeffs), p=coeffs)
    exp_pauli = paulis[exp_ind]

    cur_pauli = Pauli("I" * len(exp_pauli))
    # Sampling the first `k` terms for the product.
    k_inds = np.random.choice(len(coeffs), p=coeffs, size=k)
    for ind in k_inds:
        cur_pauli = cur_pauli @ paulis[ind]

    if alpha < 0:
        cur_pauli = -1 * cur_pauli

    exp_pauli = exp_pauli.to_matrix()
    rotated = calculate_exp(t_bar, exp_pauli, k)

    cur_pauli = cur_pauli.to_matrix()
    cur_pauli = cur_pauli @ rotated
    return cur_pauli


def sample_lcu(
    paulis: list[Pauli],
    norm_coeffs: NDArray,
    t_bar: float,
    r: int,
    error: float,
):
    # TODO obs_norm
    obs_norm = 1
    cap_k = get_cap_k(t_bar, obs_norm=obs_norm, eps=error)

    alphas = np.array(get_alphas(t_bar, cap_k, r))
    k_probs = np.array([abs(alpha) for alpha in alphas])
    # TODO: What to do of this?
    k_probs /= np.sum(k_probs)
    k = np.random.choice(cap_k + 1, p=k_probs)

    return sample_for_fixed_k(paulis, norm_coeffs, t_bar, r, k, alphas[k])


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

    alphas = np.array(get_alphas(t_bar, cap_k, r))
    k_probs = np.array([abs(alpha) for alpha in alphas])
    # TODO: What to do of this?
    k_probs /= np.sum(k_probs)

    def get_unitary(k):
        return sample_for_fixed_k(paulis, coeffs, t_bar, r, k, alphas[k])

    def control_unitary(k, control_val: int):
        """
        Calculates |0><0| U + |1><1| I if control_val = 0
        Calculates |0><0| I + |1><1| U if control_val = 1
        """
        return control_version(get_unitary(k), control_val)

    def post_v1(k):
        final_psi = psi_init.copy()

        for _ in range(r):
            v1 = control_unitary(k, control_val=1)
            final_psi = v1 @ final_psi

        npt.assert_almost_equal(np.sum(np.abs(final_psi) ** 2), 1)
        return final_psi

    def post_v1v2(k1: int, k2: int):
        final_psi = post_v1(k1)

        for _ in range(r):
            v2 = control_unitary(k2, control_val=0)
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

    with tqdm(total=count) as pbar:
        for k1, k2 in sorted(sample_ks.keys()):
            k_count = sample_ks[(k1, k2)]
            total_count += k_count

            for _ in range(k_count):
                final_psi = post_v1v2(k1, k2)
                final_rho = np.outer(final_psi, final_psi.conj())

                result = np.trace(np.abs(obs @ final_rho))
                pbar.update(1)
                results.append(result)

    assert total_count == count

    magn_h = calculate_mu(results, count, [1])
    return magn_h


class TaylorSample:
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
        self.beta = np.sum(coeffs)
        self.norm_coeffs = self.coeffs / self.beta

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

    def sample_from_lcu(self, time, psi_init, obs, control_val):
        if self.ham is None:
            raise ValueError("Parameter not substituted.")
        if self.obs is None:
            raise ValueError("Observable not substituted.")

        t_bar = time * self.beta
        # For t_bar < 1, r is too small to get accurate results.
        r = max(20, int(np.ceil(t_bar) ** 2))

        psi_final = psi_init.copy()

        print(f"Sampling:{r} terms")
        for _ in range(r):
            unitary = control_version(
                sample_lcu(self.paulis, self.norm_coeffs, t_bar, r, self.error),
                control_val,
            )
            psi_final = unitary @ psi_final

        final_rho = np.outer(psi_final, psi_final.conj())
        return np.trace(np.abs(obs @ final_rho))

    def get_observations(self, psi_init: NDArray, times: list[float]):
        return [self.get_observation(time, psi_init) for time in times]
