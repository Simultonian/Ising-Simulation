from typing import Optional
import numpy as np
from numpy.typing import NDArray

from qiskit.quantum_info import Pauli
from qiskit.circuit import Parameter

from ising.hamiltonian import Hamiltonian
from ising.hamiltonian.hamiltonian import substitute_parameter
from itertools import product as cartesian_product
import math



def get_small_k_probs(t_bar, r, cap_k):
    ks = np.arange(cap_k+1)
    k_vec = np.zeros(cap_k+1, dtype=np.complex128)

    def apply_k(k):
        # Function according to the formula
        t1 = ((1j*t_bar/r) ** k) / math.factorial(k)
        t2 = np.sqrt(1 + ((t_bar / (r * (k + 1))) ** 2))
        return t1 * t2

    vectorized_apply_k = np.vectorize(apply_k)

    k_vec = vectorized_apply_k(ks)

    # Odd positions are 0
    k_vec[1::2] = 0
    return k_vec

def get_cap_k(t_bar, obs_norm, eps) -> int:
    numr = np.log(t_bar * obs_norm / eps)
    return int(np.ceil(numr / np.log(numr)))


def normalize_ham_list(pauli_map: dict[Pauli, complex]) -> tuple[list[Pauli], list[float], complex]:
    paulis = []
    coeffs = []
    for (p, v) in pauli_map.items():
        if v.real < 0:
            paulis.append(-p)
            coeffs.append(-v)
        else:
            paulis.append(p)
            coeffs.append(v)

    coeffs = np.array(coeffs).real
    coeff_sum = np.sum(coeffs)
    coeffs /= coeff_sum

    return (paulis, coeffs, coeff_sum)

def calculate_exp_pauli(t_bar:float, r: int, k: int, pauli: NDArray) -> NDArray:
    eye = np.identity(pauli.shape[0])
    dr = np.sqrt(1 + (((t_bar/r) / (k + 1)) ** 2))

    term2 = (1j * (t_bar/r) * pauli) / (k + 1)
    rotate = (eye - term2) / dr
    return rotate

def calculate_exp(time, pauli, k):
    eye = np.identity(pauli.shape[0])
    dr = np.sqrt(1 + ((time) / (k + 1)) ** 2)

    term2 = (1j * (time) * pauli) / (k + 1)
    rotate = (eye - term2) / dr
    return rotate

def get_alphas(t_bar, cap_k, r=None):
    if r is None:
        r = np.ceil(t_bar ** 2)

    return get_small_k_probs(t_bar=t_bar, r=r, cap_k=cap_k)


def calculate_decomposition_term(prod_inds, rotation_ind, paulis, t_bar, k, alpha):
    exp_pair = paulis[rotation_ind]

    cur_prob = alpha
    cur_pauli = Pauli("I" * len(exp_pair[0]))
    for ind in prod_inds:
        pauli, prob = paulis[ind]
        cur_prob *= prob
        cur_pauli = cur_pauli @ pauli

    exp_pauli, exp_prob = exp_pair

    exp_pauli = exp_pauli.to_matrix()
    rotated = calculate_exp(t_bar, exp_pauli, k)

    cur_prob *= exp_prob
    cur_pauli = cur_pauli.to_matrix()
    cur_pauli = cur_pauli @ rotated

    assert cur_pauli is not None
    if cur_prob < 0:
        cur_prob *= -1
        cur_pauli *= -1

    return (cur_pauli, cur_prob)

def sum_decomposition_terms(paulis, time, r, coeffs, k_max):
    pairs = list(zip(paulis, coeffs))
    inds = np.arange(len(pairs))
    terms = []
    probs = []

    if r is None:
        r = int(np.ceil(time ** 2))

    alphas = get_alphas(time, k_max, r)
    t_bar = time / r

    for k in range(0, k_max+1, 2):
        alpha_term = alphas[k]

        if t_bar == 0.0:
            if k > 0:
                assert alpha_term == 0.0

        mult_inds = cartesian_product(inds, repeat=k+1)

        for inds in mult_inds:
            prod_inds, rotation_ind = inds[:-1], inds[-1]
            cur_pauli, cur_prob = calculate_decomposition_term(prod_inds, rotation_ind, paulis, t_bar, k, alpha_term)
            terms.append(cur_pauli)
            probs.append(cur_prob)

    _probs = []
    for prob in probs:
        np.testing.assert_allclose(prob.imag, 0.0)
        _probs.append(prob.real)

    return (terms, np.array(_probs))

from functools import lru_cache
import numpy as np
import numpy.testing as npt
from collections import Counter

from qiskit.quantum_info import SparsePauliOp

from ising.hamiltonian import Hamiltonian
from ising.utils.constants import ONEONE, ZEROZERO, PLUS
from ising.utils import MAXSIZE
from ising.singlelcu.simulation.singlelcu import calculate_mu


class SingleAncilla:
    def __init__(self, terms, probs, observable: Hamiltonian, **kwargs):
        """
        Code for running single ancilla LCU simulation using given synthesizer.

        Inputs:
            - synthesizer: Provides access for construction of unitaries.
            - observable: Observable to run LCU simulation on.

        Kwargs must contain:
            - eeta
            - eps
            - prob
        """
        self.terms = terms
        self.probs = probs
        obs_x = SparsePauliOp(["X"], [1.0])
        run_obs = obs_x.tensor(observable.sparse_repr)

        self.run_obs = run_obs.to_matrix()

        # TODO: Replace with `spectral_norm`
        self.obs_norm = 1

        self.params = {x: kwargs[x] for x in ["eeta", "eps", "prob"]}

        self.inds = list(range(len(terms)))
        self.eye = np.identity(terms[0].shape[0])

        self.rho_init = None

    def get_unitary(self, ind: int):
        """
        Uses the synthesizer for constructing Hamiltonian simulation for given
        lcu time index.

        Inputs:
            - ind: Sampled index
        """
        return self.terms[ind]

    def control_unitary(self, ind: int, control_val: int):
        """
        Calculates |0><0| U + |1><1| I if control_val = 0
        Calculates |0><0| I + |1><1| U if control_val = 1

        Where U is unitary for Hamiltonian evolution with time `times[ind]`.
        """

        unitary = self.get_unitary(ind)

        if control_val == 0:
            op_1 = np.kron(ZEROZERO, unitary)
            op_2 = np.kron(ONEONE, self.eye)
            return op_1 + op_2
        # Control value is 1
        else:
            op_1 = np.kron(ZEROZERO, self.eye)
            op_2 = np.kron(ONEONE, unitary)
            return op_1 + op_2

    @lru_cache(maxsize=MAXSIZE)
    def post_v1(self, ind):
        if self.rho_init is None:
            raise ValueError("Initial state not set.")
        final_rho = self.rho_init.copy()
        v1 = self.control_unitary(ind, control_val=1)
        final_rho = v1 @ final_rho @ v1.conj().T
        npt.assert_allclose(np.trace(final_rho), 1, atol=1e-3, rtol=1e-3)
        return final_rho

    def post_v1v2(self, i1, i2):
        final_rho = self.post_v1(i1)

        v2 = self.control_unitary(i2, control_val=0)
        final_rho = v2 @ final_rho @ v2.conj().T

        npt.assert_allclose(np.trace(final_rho), 1, atol=1e-3, rtol=1e-3)

        return final_rho

    def calculate_mu(self, count):
        print("Entering loop")
        results = []

        samples_count = Counter(
            [tuple(x) for x in np.random.choice(self.inds, p=self.probs, size=(count, 2))]
        )

        total_count = 0

        for sample in sorted(samples_count.keys()):
            s_count = samples_count[sample]
            if total_count % 100 == 0:
                print(f"running: {total_count} out of {count}")
            total_count += s_count

            final_rho = self.post_v1v2(sample[0], sample[1])

            result = np.trace(np.abs(self.run_obs @ final_rho))
            results.append(result * s_count)

        assert total_count == count

        magn_h = calculate_mu(results, count, self.probs)
        return magn_h


class TaylorBad:
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
        self.paulis = self.ham_subbed.paulis
        self.coeffs = self.ham_subbed.coeffs
        self.beta = np.sum(np.abs(np.array(self.coeffs)))
        self.coeffs /= self.beta

    def construct_parametrized_circuit(self) -> None:
        if self.ham_subbed is None:
            raise ValueError(
                "h value has not been substituted, qiskit does not support parametrized Hamiltonians."
            )

    def get_observation(self, rho_init: NDArray, observable: Hamiltonian, time: float):
        self.r = np.ceil(time ** 2)
        self.t_bar = time / self.beta

        # TODO obs_norm
        self.cap_k = get_cap_k(self.t_bar, obs_norm=1, eps=self.error)
        self.terms, self.probs = sum_decomposition_terms(self.paulis, time, self.r, self.coeffs, self.cap_k)

        c_1 = np.sum(np.abs(self.probs))
        self.probs /= c_1

        init_complete = np.kron(PLUS, rho_init)

        T = 1000



    def get_observations(
        self, rho_init: NDArray, observable: Hamiltonian, times: list[float]
    ):
        results = []
        for time in times:
            unitary = self.matrix(time)
            rho_final = unitary @ rho_init @ unitary.conj().T
            result = np.trace(np.abs(observable @ rho_final))
            results.append(result)

        return results
