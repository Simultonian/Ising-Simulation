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
from ising.utils.constants import ONEONE, ZEROZERO
from ising.singlelcu.simulation.singlelcu import calculate_mu

from ising.simulation.taylor.taylor import get_cap_k, calculate_exp, get_alphas


def calculate_decomposition_term(prod_inds, rotation_ind, paulis, t_bar, k):
    exp_pair = paulis[rotation_ind]

    cur_prob = 1.0
    cur_pauli = Pauli("I" * len(exp_pair[0]))
    for ind in prod_inds:
        pauli, prob = paulis[ind]
        cur_prob *= prob
        cur_pauli = cur_pauli @ pauli

    exp_pauli, exp_prob = exp_pair
    if exp_prob < 0:
        exp_pauli = exp_pauli * -1
        exp_prob *= -1

    exp_pauli = exp_pauli.to_matrix()
    rotated = calculate_exp(t_bar, exp_pauli, k)

    assert exp_prob >= 0
    cur_prob *= exp_prob
    cur_pauli = cur_pauli.to_matrix()
    cur_pauli = cur_pauli @ rotated

    assert cur_pauli is not None
    if cur_prob < 0:
        cur_prob *= -1
        cur_pauli *= -1

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

        # REMOVE
        # mult_inds = cartesian_product(inds, repeat=k + 1)
        terms = []
        probs = []

        # for term_inds in mult_inds:
        #     prod_inds, rotation_ind = term_inds[:-1], term_inds[-1]

        #     cur_pauli, cur_prob = calculate_decomposition_term(
        #         prod_inds, rotation_ind, pairs, t_bar, k
        #     )

        #     if alpha_term < 0:
        #         # Transferring the alpha negative to the term.
        #         cur_pauli *= -1

        #     assert isinstance(cur_pauli, np.ndarray)
        #     terms.append(cur_pauli)
        #     probs.append(cur_prob)

        # probs = np.array(probs)
        # probs = probs.real
        probs = [1]
        npt.assert_allclose(np.sum(probs), 1, atol=1e-5, rtol=1e-5)

        k_probs.append(abs(alpha_term))

        kth_probs.append(probs)
        kth_paulis.append(terms)

    assert len(k_probs) == len(kth_probs)
    assert len(k_probs) == len(kth_paulis)

    k_probs = np.array(k_probs)
    k_probs /= np.sum(k_probs)

    return (kth_paulis, kth_probs, k_probs)


def taylor_observation(
    ham: Hamiltonian, time: float, error: float, obs, rho_init, **kwargs
):
    delta = 1 - kwargs.get("success", 0.9)
    paulis = ham.paulis
    coeffs = ham.coeffs
    beta = np.sum(np.abs(np.array(coeffs)))
    coeffs /= beta

    t_bar = time * beta
    r = int(5 * np.ceil(t_bar) ** 2)

    # For t_bar < 1, r is too small to get accurate results.
    r = max(20, r)

    # TODO obs_norm
    obs_norm = 1
    cap_k = get_cap_k(t_bar, obs_norm=obs_norm, eps=error)

    print(f"Computing decomposition for t_bar={t_bar} t={time} r={r} k={cap_k}")
    kth_paulis, kth_probs, k_probs = sum_decomposition_k_fold(
        paulis, t_bar, r, coeffs, cap_k
    )
    print("Decomposition complete")

    # eye = np.identity(kth_paulis[0][0].shape[0])
    eye = np.identity(obs.shape[0] - 1)

    def get_unitary(k, ind: int):
        return kth_paulis[k][ind]

    def control_unitary(k, ind: int, control_val: int):
        """
        Calculates |0><0| U + |1><1| I if control_val = 0
        Calculates |0><0| I + |1><1| U if control_val = 1

        Where U is unitary for Hamiltonian evolution with time `times[ind]`.
        """

        unitary = get_unitary(k, ind)

        if control_val == 0:
            op_1 = np.kron(ZEROZERO, unitary)
            op_2 = np.kron(ONEONE, eye)
            return op_1 + op_2
        # Control value is 1
        else:
            op_1 = np.kron(ZEROZERO, eye)
            op_2 = np.kron(ONEONE, unitary)
            return op_1 + op_2

    def post_v1(k, inds: list[int]):
        if rho_init is None:
            raise ValueError("Initial state not set.")

        final_rho = rho_init.copy()

        for ind in inds:
            v1 = control_unitary(k, ind, control_val=1)
            final_rho = v1 @ final_rho @ v1.conj().T

        npt.assert_allclose(np.trace(final_rho), 1, atol=1e-3, rtol=1e-3)
        return final_rho

    def post_v1v2(ks: tuple[int, int], i1s: list[int], i2s: list[int]):
        final_rho = post_v1(ks[0], i1s)

        for i2 in i2s:
            v2 = control_unitary(ks[1], i2, control_val=0)
            final_rho = v2 @ final_rho @ v2.conj().T

        npt.assert_allclose(np.trace(final_rho), 1, atol=1e-3, rtol=1e-3)

        return final_rho

    print("Entering loop of Taylor Single")

    total_count = 0
    count = 1000
    count = int(8 * np.ceil(((obs_norm**2) * (np.log(2 / delta))) / (error**2)))
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

        # Two extra for rotation
        run_cost = (k1 + k2 + 2) * r
        print(f"Run cost: {run_cost}")

        k1_terms = Counter(
            [
                tuple(x)
                for x in np.random.choice(
                    len(kth_probs[k1]), p=kth_probs[k1], size=(k_count, r)
                )
            ]
        )

        k2_terms = Counter(
            [
                tuple(x)
                for x in np.random.choice(
                    len(kth_probs[k2]), p=kth_probs[k2], size=(k_count, r)
                )
            ]
        )

        for k1_term, k2_term in zip(k1_terms, k2_terms):
            # REMOVE
            # final_rho = post_v1v2((k1, k2), list(k1_term), list(k2_term))
            # result = np.trace(np.abs(obs @ final_rho))

            result = 0
            results.append(result)

    assert total_count == count

    magn_h = calculate_mu(results, count, [1])
    return magn_h


class TaylorSingle:
    """
    Creates the entire decomposition and then samples from that.
    """

    def __init__(self, ham: Hamiltonian, h: Parameter, error: float, **kwargs):
        self.ham = ham
        self.num_qubits = ham.sparse_repr.num_qubits
        self.error = error
        self.ham_subbed: Optional[Hamiltonian] = None

        self.h = h
        self.success = kwargs.get("success", 0.9)

    @property
    def ground_state(self) -> NDArray:
        if self.ham_subbed is None:
            raise ValueError(
                "h value has not been substituted, qiskit does not support parametrized Hamiltonians."
            )
        return self.ham_subbed.ground_state

    def subsitute_h(self, h_val: float) -> None:
        self.ham_subbed = substitute_parameter(self.ham, self.h, h_val)

    def construct_parametrized_circuit(self) -> None:
        if self.ham_subbed is None:
            raise ValueError(
                "h value has not been substituted, qiskit does not support parametrized Hamiltonians."
            )

    def get_observation(self, time: float):
        if self.ham_subbed is None:
            raise ValueError("Parameter not substituted.")
        return taylor_observation(
            self.ham_subbed,
            time,
            self.error,
            self.run_obs,
            self.rho_init,
            success=self.success,
        )

    def get_observations(
        self, rho_init: NDArray, observable: Hamiltonian, times: list[float]
    ):
        results = []
        self.obs_init = observable.matrix
        obs_x = SparsePauliOp(["X"], [1.0])
        run_obs = obs_x.tensor(observable.sparse_repr)

        self._run_obs = run_obs
        self.run_obs = run_obs.to_matrix()
        self.rho_init = rho_init
        for time in times:
            result = self.get_observation(time)
            results.append(result)

        return results
