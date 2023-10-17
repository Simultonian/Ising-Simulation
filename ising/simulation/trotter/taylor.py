from typing import Optional, Union, Sequence
from functools import lru_cache, reduce
import numpy as np
from numpy.typing import NDArray

from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.synthesis import LieTrotter

from ising.hamiltonian import Hamiltonian, trotter_reps, general_grouping
from ising.hamiltonian.hamiltonian import substitute_parameter
from ising.utils import MAXSIZE
from ising.utils import simdiag
from itertools import product as cartesian_product



def get_small_k_probs(t_bar, r, cap_k):
    ks = np.arange(cap_k+1)
    k_vec = np.zeros(cap_k+1, dtype=np.complex128)

    def apply_k(k):
        # Function according to the formula
        t1 = ((1j*t_bar/r) ** k) / np.math.factorial(k)
        t2 = np.sqrt(1 + ((t_bar / (r * k + 1)) ** 2))
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

def get_final_term_from_sample(indices, rotation_ind, paulis, normalized_ham_coeffs, alpha, t_bar, r, k):
    assert len(indices) == k
    if k > 0:
        coeff_prod = np.prod([normalized_ham_coeffs[ind] for ind in indices])
        pauli_prod = Pauli("I" * len(paulis[rotation_ind]))
        for ind in indices:
            pauli_prod = pauli_prod @ paulis[ind]
    else:
        coeff_prod = 1
        pauli_prod = Pauli("I" * len(paulis[rotation_ind]))

    # Do not multiply it for the phase because it is pushed in exp
    coeff_prod *= alpha

    rotation_pauli_mat = paulis[rotation_ind].to_matrix()

    # Taking care of negative sign in sampled rotation
    if normalized_ham_coeffs[rotation_ind] < 0:
        rotation_pauli_mat *= -1

    exp_pauli = calculate_exp_pauli(t_bar, r, k, rotation_pauli_mat)

    phase = 1
    if coeff_prod < 0:
        phase = -1

    return phase * (pauli_prod.to_matrix() @ exp_pauli)


def calculate_exp(time, pauli, k):
    eye = np.identity(pauli.shape[0])
    dr = np.sqrt(1 + ((time) / (k + 1)) ** 2)

    term2 = (1j * (time) * pauli) / (k + 1)
    rotate = (eye - term2) / dr
    return rotate

def sum_decomposition(paulis, time, coeffs, k_max, alphas):
    pairs = list(zip(paulis, coeffs))
    final = None

    for k in range(0, k_max+1, 2):
        alpha_term = alphas[k]

        if time == 0.0:
            if k > 0:
                assert alpha_term == 0.0

        mult_paulis = cartesian_product(pairs, repeat=k+1)

        total_pauli = None
        for paulis in mult_paulis:
            pauli_prod = paulis[:-1]
            exp_pair = paulis[-1]

            cur_prob = 1.0
            cur_pauli = Pauli("I" * len(exp_pair[0]))
            for pauli, prob in pauli_prod:
                cur_prob *= prob
                cur_pauli = cur_pauli @ pauli

            exp_pauli, exp_prob = exp_pair
            exp_pauli = exp_pauli.to_matrix()
            rotated = calculate_exp(time, exp_pauli, k)

            cur_pauli = cur_prob * cur_pauli.to_matrix()
            cur_pauli = cur_pauli @ (exp_prob * rotated)

            assert cur_pauli is not None

            if total_pauli is None:
                total_pauli = cur_pauli
            else:
                total_pauli += cur_pauli

        assert total_pauli is not None
        total_pauli *= alpha_term

        if final is None:
            final = total_pauli
        else:
            final += total_pauli
        
    return final



class Taylor:
    def __init__(self, ham: Hamiltonian, h: Parameter, error: float, delta: float):
        self.ham = ham
        self.num_qubits = ham.sparse_repr.num_qubits
        self.error = error
        self.delta = delta
        self.ham_subbed: Optional[Hamiltonian] = None

        self.h = h
        self.time = Parameter("t")

        self.paulis = ham.paulis

        # TODO: Remove this
        self.obs_norm = 1

    @property
    def ground_state(self) -> NDArray:
        if self.ham_subbed is None:
            raise ValueError(
                "h value has not been substituted, qiskit does not support parametrized Hamiltonians."
            )
        return self.ham_subbed.ground_state

    def subsitute_h(self, h_val: float) -> None:
        self.ham_subbed = substitute_parameter(self.ham, self.h, h_val)

        self.paulis, self.coeffs, self.beta = normalize_ham_list(self.ham_subbed.map)
        self.pauli_inds = np.arange(len(self.paulis))


    def construct_parametrized_circuit(self) -> None:
        if self.ham_subbed is None:
            raise ValueError(
                "h value has not been substituted, qiskit does not support parametrized Hamiltonians."
            )


    def get_exact_unitary(self, time):
        assert self.ham_subbed is not None
        return self.ham_subbed.eig_vec @ np.diag(np.exp(complex(0, -1) * time * self.ham_subbed.eig_val)) @ self.ham_subbed.eig_vec_inv 


    def get_alphas(self, time:float):
        t_bar = time * self.beta
        r = np.ceil(t_bar ** 2)
        # cap_k = get_cap_k(t_bar, self.obs_norm, self.error)
        cap_k = 3

        return get_small_k_probs(t_bar=t_bar, r=r, cap_k=cap_k)

    def sample_v(self, time:float):

        t_bar = time * self.beta
        r = int(np.ceil(t_bar ** 2))
        # cap_k = get_cap_k(t_bar, self.obs_norm, self.error)


        # TODO
        cap_k = 3
        r = 1

        alphas = get_small_k_probs(t_bar=t_bar, r=r, cap_k=cap_k)

        k_probs = np.abs(alphas)
        k_probs /= np.sum(k_probs)
        k_range = np.arange(0,cap_k+1)

        assert k_range.shape == k_probs.shape

        final = None
        for _ in range(r):
            
            # Sample k
            k_cur = np.random.choice(k_range, p=k_probs)
            alpha_cur = alphas[k_cur]
            sampled_pauli_inds = np.random.choice(self.pauli_inds, p=self.coeffs, size=k_cur)
            rotation_ind = np.random.choice(self.pauli_inds, p=self.coeffs, size=1)[0]

            cur_term = get_final_term_from_sample(sampled_pauli_inds, rotation_ind, self.paulis, self.coeffs, alpha_cur, t_bar, r, k_cur)

            if final is None:
                final = cur_term
            else:
                final = final @ cur_term

        
        return final

    def get_observation(self, rho_init: NDArray, observable: NDArray, time:float):
        """
        Run the singleLCU algorithm and get the observation value for the simulation.

        e^{-iHt} R e^{iHt}
        """

        ## Previous
        unitary = self.matrix(time)
        rho_final = unitary @ rho_init @ unitary.conj().T
        result = np.trace(np.abs(observable @ rho_final))
        return result

    def get_observations(
        self, rho_init: NDArray, observable: NDArray, times: list[float]
    ):
        # TODO
        self.obs_norm = 1
        results = []
        for time in times:
            result = self.get_observation(rho_init, observable, time)
            results.append(result)

        return results
