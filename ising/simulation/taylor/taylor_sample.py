from typing import Optional
import numpy as np
from numpy.typing import NDArray

from qiskit.quantum_info import Pauli
from qiskit.circuit import Parameter

from ising.hamiltonian import Hamiltonian
from ising.hamiltonian.hamiltonian import substitute_parameter

import numpy as np
import numpy.testing as npt

from qiskit.quantum_info import SparsePauliOp

from ising.hamiltonian import Hamiltonian
from ising.utils.constants import ONEONE, ZEROZERO
from ising.singlelcu.simulation.singlelcu import calculate_mu

from ising.simulation.taylor.taylor import calculate_exp, get_cap_k, get_alphas


def calculate_decomposition_term(prod_inds, rotation_ind, paulis, t_bar, k):
    exp_pauli = paulis[rotation_ind]

    cur_pauli = Pauli("I" * len(exp_pauli))
    for ind in prod_inds:
        pauli = paulis[ind]
        cur_pauli = cur_pauli @ pauli

    exp_pauli = exp_pauli.to_matrix()
    rotated = calculate_exp(t_bar, exp_pauli, k)

    cur_pauli = cur_pauli.to_matrix()
    cur_pauli = cur_pauli @ rotated

    assert cur_pauli is not None

    return cur_pauli


def sample_inner_term(paulis, coeffs, k, t_bar):
    # Sample k + 1 terms

    pauli_count = len(paulis)
    prod_terms = np.random.choice(pauli_count, size=k, p=coeffs)
    rotation_term = np.random.choice(pauli_count, size=1, p=coeffs)[0]

    return calculate_decomposition_term(prod_terms, rotation_term, paulis, t_bar, k)


def is_unitary(a):
    return np.allclose(np.dot(a, np.conj(a.T)), np.eye(a.shape[0]))


class TaylorSample:
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

        paulis = self.ham_subbed.paulis
        coeffs = self.ham_subbed.coeffs

        self.paulis = []
        self.coeffs = []
        for pauli, _coeff in zip(paulis, coeffs):
            coeff = _coeff.real
            assert coeff.imag == 0

            if coeff > 0:
                self.paulis.append(pauli)
                self.coeffs.append(coeff)
            if coeff < 0:
                self.paulis.append(-pauli)
                self.coeffs.append(-coeff)

        self.beta = np.sum(self.coeffs)
        self.coeffs = np.array(self.coeffs) / self.beta

    def construct_parametrized_circuit(self) -> None:
        """
        This method is called to construct the LCU for the truncated Taylor
        series.
        """
        if self.ham_subbed is None:
            raise ValueError(
                "h value has not been substituted, qiskit does not support parametrized Hamiltonians."
            )

    def get_unitary(self):
        """
        Uses the synthesizer for constructing Hamiltonian simulation for given
        lcu time index.

        Inputs:
            - ind: Sampled index
        """
        k = np.random.choice(self.cap_k + 1, size=1, p=self.alphas)
        assert k % 2 == 0
        if (k // 2) % 2 == 0:
            unitary = -sample_inner_term(self.paulis, self.probs, k, self.t_bar)
        else:
            # (-i) ** k becomes -1
            unitary = sample_inner_term(self.paulis, self.probs, k, self.t_bar)

        assert is_unitary(unitary)

        return unitary

    def control_unitary(self, control_val: int):
        """
        Calculates |0><0| U + |1><1| I if control_val = 0
        Calculates |0><0| I + |1><1| U if control_val = 1

        Where U is unitary for Hamiltonian evolution with time `times[ind]`.
        """

        unitary = self.get_unitary()

        if control_val == 0:
            op_1 = np.kron(ZEROZERO, unitary)
            op_2 = np.kron(ONEONE, self.eye)
            return op_1 + op_2
        # Control value is 1
        else:
            op_1 = np.kron(ZEROZERO, self.eye)
            op_2 = np.kron(ONEONE, unitary)
            return op_1 + op_2

    def post_v1(self):
        if self.rho_init is None:
            raise ValueError("Initial state not set.")

        final_rho = self.rho_init.copy()

        for _ in range(self.r):
            v1 = self.control_unitary(control_val=1)
            final_rho = v1 @ final_rho @ v1.conj().T

        npt.assert_allclose(np.trace(final_rho), 1, atol=1e-3, rtol=1e-3)
        return final_rho

    def post_v1v2(self):
        final_rho = self.post_v1()

        for _ in range(self.r):
            v2 = self.control_unitary(control_val=0)
            final_rho = v2 @ final_rho @ v2.conj().T

        npt.assert_allclose(np.trace(final_rho), 1, atol=1e-3, rtol=1e-3)

        return final_rho

    def get_observation(self, time: float):
        self.time = time
        self.t_bar = time * self.beta
        self.r = int(5 * np.ceil(self.t_bar) ** 2)

        # TODO obs_norm
        self.cap_k = get_cap_k(self.t_bar, obs_norm=1, eps=self.error)

        self.cap_k = 1
        self.r = 4

        self.eye = np.identity(2 ** len(self.paulis[0]))
        alphas = get_alphas(self.t_bar, self.cap_k, self.r)

        self.alphas = []
        for alpha in alphas:
            assert alpha.imag == 0
            self.alphas.append(abs(alpha.real))

        self.alpha_sum = sum(self.alphas)
        self.alphas = np.array(self.alphas) / self.alpha_sum

        c_1 = np.sum(self.coeffs)
        self.probs = self.coeffs / c_1

        count = 1000
        results = []

        print(f"r:{self.r} k:{self.cap_k}")
        print("Entering loop")

        total_count = 0

        for _ in range(count):
            if total_count % 100 == 0:
                print(f"running: {total_count} out of {count}")
            total_count += 1

            final_rho = self.post_v1v2()

            result = np.trace(np.abs(self.run_obs @ final_rho))
            results.append(result)

        assert total_count == count

        magn_h = calculate_mu(results, count, [1])
        return magn_h

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
