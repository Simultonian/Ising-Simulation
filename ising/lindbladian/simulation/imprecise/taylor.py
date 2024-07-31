from typing import Optional
from collections import defaultdict
from functools import lru_cache
import numpy as np
from numpy.typing import NDArray

from qiskit.quantum_info import PauliList
from qiskit.circuit import Parameter
from ising.utils.trace import partial_trace

from ising.hamiltonian import Hamiltonian
from ising.hamiltonian.hamiltonian import substitute_parameter
from itertools import product as cartesian_product

import numpy as np
import numpy.testing as npt
from collections import Counter

from qiskit.quantum_info import SparsePauliOp, Pauli

from ising.hamiltonian import Hamiltonian
from ising.utils import MAXSIZE, control_version
from ising.simulation.taylor.utils import (
    get_cap_k,
    calculate_exp,
    get_alphas,
    calculate_mu,
)

from ising.utils import global_phase
from tqdm import tqdm

PSI_PLUS = np.array([[1], [1]]) / np.sqrt(2)
RHO_PLUS = np.outer(PSI_PLUS, PSI_PLUS.conj())

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

@lru_cache(maxsize=None)
def pauli_to_matrix(pauli):
    return pauli.to_matrix()


class Taylor:
    """
    Creates the entire decomposition and then samples from that.
    """

    def __init__(self, sent_paulis: list[Pauli], sent_coeffs: list[float], error: float, **kwargs):
        paulis, coeffs = [], []

        for pauli, _coeff in zip(sent_paulis, sent_coeffs):
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

        self.num_qubits = paulis[0].num_qubits
        self.error = error

        self.success = kwargs.get("success", 0.9)

    def get_k_terms(self, k, count, r):
        return Counter(
            [
                tuple(x)
                for x in np.random.choice(
                    len(self.kth_probs[k]), p=self.kth_probs[k], size=(count, r)
                )
            ]
        )

    def get_unitary(self, time: float, k: int, ind: int):
        rotated = calculate_exp(time, pauli_to_matrix(self.kth_exps[k][ind]), k)
        return pauli_to_matrix(self.kth_paulis[k][ind]) @ rotated

    def control_unitary(self, time: float, k, ind: int, control_val: int):
        """
        Calculates |0><0| U + |1><1| I if control_val = 0
        Calculates |0><0| I + |1><1| U if control_val = 1
        """
        return control_version(self.get_unitary(time, k, ind), control_val)

    def post_v1(self, rho_init, time: float, k, inds: tuple[int, ...]):
        final_rho = rho_init.copy() 

        for ind in inds:
            v1 = self.control_unitary(time, k, ind, control_val=1)
            final_rho = v1 @ final_rho @ v1.conj().T

        assert np.isclose(np.trace(final_rho), 1)
        return final_rho

    def post_v1v2(
        self,
        rho_init,
        time: float,
        ks: tuple[int, int],
        i1s: tuple[int, ...],
        i2s: tuple[int, ...],
    ):
        k1, k2 = ks
        final_rho = self.post_v1(rho_init, time, k1, i1s)

        for i2 in i2s:
            v2 = self.control_unitary(time, k2, i2, control_val=0)
            final_rho = v2 @ final_rho @ v2.conj().T

        assert np.isclose(np.trace(final_rho), 1)
        return final_rho

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

    def setup_time(self, time: float):
        """
        Set the time for which SAL will be used.
        """
        self.t_bar = time * self.beta
        # TODO obs_norm
        obs_norm = 1
        self.cap_k = get_cap_k(self.t_bar, obs_norm=obs_norm, eps=self.error)

        print(f"Computing decomposition for t_bar={self.t_bar} t={time} k={self.cap_k}")
        self.kth_paulis, self.kth_exps, self.kth_probs = sum_decomposition_k_fold(
            self.paulis, self.coeffs, self.cap_k
        )
        print("Decomposition complete")

        # For t_bar < 1, r is too small to get accurate results.
        self.r = max(20, int(5 * np.ceil(self.t_bar) ** 2))
        self.evo_time = np.round(self.t_bar / self.r, 6)

        self.alphas = get_alphas(self.t_bar, self.cap_k, self.r)
        k_probs = np.abs(self.alphas)
        self.k_probs = k_probs / np.sum(k_probs)

    def sample_and_evolve(self, rho_init):
        """
        Given rho_init we evolve the state by time t using SAL after freshly 
        sampling the unitaries. Returns the new density matrix
        """
        final_rho = np.kron(
            RHO_PLUS, rho_init
        )
        neg = 1
        for _ in range(self.r):
            k_1, k_2 = np.random.choice(self.cap_k + 1, p=self.k_probs, size=2)

            k1_term = np.random.choice(len(self.kth_probs[k_1]), p=self.kth_probs[k_1])
            v1 = self.control_unitary(self.evo_time, k_1, k1_term, control_val=1)
            final_rho = v1 @ final_rho @ v1.conj().T

            k2_term = np.random.choice(len(self.kth_probs[k_2]), p=self.kth_probs[k_2])

            v2 = self.control_unitary(self.evo_time, k_2, k2_term, control_val=1)
            final_rho = v2 @ final_rho @ v2.conj().T

            if self.alphas[k_1] < 0 and self.r % 2 == 1:
                neg *= -1

            if self.alphas[k_2] < 0 and self.r % 2 == 1:
                neg *= -1

        final_rho = neg * final_rho
        final_rho = partial_trace(final_rho, [0])
        return final_rho

def collision_model_evo(
    rho_sys: NDArray,
    big_rho_env: NDArray,
    ham_ints: list[tuple[list[Pauli], list[float]]],
    ham_sys: tuple[list[Pauli], list[float]],
    tau: float,
    error: float,
    neu: int,
    runs: int,
    observable,
):
    """
    Replicate the Lindbladian evolution of amplitude damping using
    interaction Hamiltonian dynamics.

    Inputs:
        - rho_sys: Initial state of system only
        - rho_env: Initial state of environment only
        - ham_ints: Interaction Hamiltonians in Pauli basis (gamma included)
        - ham_sys: Pauli basis representation of the system Hamiltonian
        - gamma: Strength of amplitude damping
        - time: Evolution time to match
    """

    sys_paulis, sys_coeffs = ham_sys

    if len(sys_paulis) == 0 or len(sys_paulis) != len(sys_coeffs):
        raise ValueError("System Hamiltonian Incorrect Dimension")

    qubit_count: int = sys_paulis[0].num_qubits // 2


    print(f"Calculating Hamiltonians for tau:{tau}, neu:{neu}")
    final_hams = []
    for (int_paulis, int_coeffs) in ham_ints:
        ham_int = defaultdict(float)

        # Adding the system Hamiltonian part
        for (sys_pauli, sys_coeff) in zip(sys_paulis, sys_coeffs):
            ham_int[sys_pauli.to_label()] = (np.sqrt(tau) * sys_coeff / qubit_count)

        # Adding the interaction Hamiltonian part
        for (int_pauli, int_coeff) in zip(int_paulis, int_coeffs):
            ham_int[int_pauli.to_label()] += int_coeff

        final_paulis, final_coeffs = [], []
        for pauli, coeff in ham_int.items():
            final_paulis.append(pauli)
            final_coeffs.append(coeff)
        
        final_hams.append((final_paulis, final_coeffs))

    print("Ham operator calculation complete")

    taylors = []
    for (paulis, coeffs) in final_hams:
        pauli_objs = [Pauli(pauli) for pauli in paulis]
        taylor = Taylor(pauli_objs, coeffs, error)
        taylor.setup_time(tau)
        taylors.append(taylor)

    results = []
    with tqdm(total=runs) as pbar:
        for _ in range(runs):
            cur_rho_sys = rho_sys
            pbar.update(1)

            for _ in range(neu):
                for taylor in taylors:
                    complete_rho = np.kron(cur_rho_sys, big_rho_env)
                    rho_fin = taylor.sample_and_evolve(complete_rho)
                    cur_rho_sys = partial_trace(
                        rho_fin, list(range(qubit_count, 2 * qubit_count))
                    )

            cur_rho_sys = cur_rho_sys / global_phase(cur_rho_sys)

            result = np.trace(np.abs(observable @ cur_rho_sys))
            results.append(result)

    magn_h = calculate_mu(results, runs, [1])
    return magn_h
