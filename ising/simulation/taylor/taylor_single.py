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


def calculate_decomposition_term(prod_inds, rotation_ind, paulis, t_bar, k, alpha):
    exp_pair = paulis[rotation_ind]

    cur_prob = alpha
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


def sum_decomposition_terms(paulis, t_bar, r, coeffs, k_max):
    pairs = list(zip(paulis, coeffs))
    pairs = list(filter(lambda x: abs(x[1]) > 0, pairs))
    inds = np.arange(len(pairs))
    terms = []
    probs = []

    alphas = get_alphas(t_bar, k_max, r)
    t_bar = t_bar / r

    for k in range(0, k_max + 1, 2):
        alpha_term = alphas[k]

        if t_bar == 0.0:
            if k > 0:
                assert alpha_term == 0.0

        mult_inds = cartesian_product(inds, repeat=k + 1)

        for inds in mult_inds:
            prod_inds, rotation_ind = inds[:-1], inds[-1]

            cur_pauli, cur_prob = calculate_decomposition_term(
                prod_inds, rotation_ind, pairs, t_bar, k, alpha_term
            )
            terms.append(cur_pauli)
            probs.append(cur_prob)

    _probs = []
    for prob in probs:
        np.testing.assert_allclose(prob.imag, 0.0)
        _probs.append(prob.real)

    return (terms, np.array(_probs))


def taylor_observation(ham: Hamiltonian, time: float, error: float, obs, rho_init):
    paulis = ham.paulis
    coeffs = ham.coeffs
    beta = np.sum(np.abs(np.array(coeffs)))
    coeffs /= beta

    t_bar = time * beta
    r = 5 * np.ceil(t_bar) ** 2
    # TODO obs_norm
    cap_k = get_cap_k(t_bar, obs_norm=1, eps=error)
    terms, probs = sum_decomposition_terms(paulis, t_bar, r, coeffs, cap_k)

    eye = np.identity(terms[0].shape[0])

    c_1 = np.sum(np.abs(probs))
    probs /= c_1
    inds = np.array(list(range(len(terms))))

    def get_unitary(ind: int):
        return terms[ind]

    def control_unitary(ind: int, control_val: int):
        """
        Calculates |0><0| U + |1><1| I if control_val = 0
        Calculates |0><0| I + |1><1| U if control_val = 1

        Where U is unitary for Hamiltonian evolution with time `times[ind]`.
        """

        unitary = get_unitary(ind)

        if control_val == 0:
            op_1 = np.kron(ZEROZERO, unitary)
            op_2 = np.kron(ONEONE, eye)
            return op_1 + op_2
        # Control value is 1
        else:
            op_1 = np.kron(ZEROZERO, eye)
            op_2 = np.kron(ONEONE, unitary)
            return op_1 + op_2

    def post_v1(inds: list[int]):
        if rho_init is None:
            raise ValueError("Initial state not set.")

        final_rho = rho_init.copy()

        for ind in inds:
            v1 = control_unitary(ind, control_val=1)
            final_rho = v1 @ final_rho @ v1.conj().T

        npt.assert_allclose(np.trace(final_rho), 1, atol=1e-3, rtol=1e-3)
        return final_rho

    def post_v1v2(i1s: list[int], i2s: list[int]):
        final_rho = post_v1(i1s)

        for i2 in i2s:
            v2 = control_unitary(i2, control_val=0)
            final_rho = v2 @ final_rho @ v2.conj().T

        npt.assert_allclose(np.trace(final_rho), 1, atol=1e-3, rtol=1e-3)

        return final_rho

    print("Entering loop of Taylor Single")

    total_count = 0
    count = 1000
    results = []
    samples_count = Counter(
        [
            tuple([tuple(x[0]), tuple(x[1])])
            for x in np.random.choice(inds, p=probs, size=(count, 2, int(r)))
        ]
    )

    for sample in sorted(samples_count.keys()):
        s_count = samples_count[sample]
        if total_count % 100 == 0:
            print(f"running: {total_count} out of {count}")
        total_count += s_count

        final_rho = post_v1v2(list(sample[0]), list(sample[1]))

        result = np.trace(np.abs(obs @ final_rho))
        results.append(result * s_count)

    assert total_count == count

    magn_h = calculate_mu(results, count, [1])
    return magn_h


class TaylorSingle:
    """
    Creates the entire decomposition and then samples from that.
    """

    def __init__(self, ham: Hamiltonian, h: Parameter, error: float):
        self.ham = ham
        self.num_qubits = ham.sparse_repr.num_qubits
        self.error = error
        self.ham_subbed: Optional[Hamiltonian] = None

        self.h = h

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
            self.ham_subbed, time, self.error, self.run_obs, self.rho_init
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
