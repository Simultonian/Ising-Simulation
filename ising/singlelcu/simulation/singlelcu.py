import numpy as np
import numpy.testing as npt
from collections import Counter
from numpy.typing import NDArray

from qiskit.quantum_info import SparsePauliOp

from ising.hamiltonian import Hamiltonian
from ising.simulation import Synthesizer
from ising.utils.constants import ONEONE, ZEROZERO, PLUS
from ising.utils import close_state


def spectral_norm(eigenval):
    """
    ||O|| = max eigenvalue of (O @ O^T), can be rewritten as the following
    """
    max_eigval = max(np.abs(eigenval**2)) ** 0.5
    return max_eigval


def calculate_mu(mu_samples, count, coeffs):
    """
    Formula is in the algorithm 1 of the paper
    """
    norm_1_sq = np.linalg.norm(coeffs, ord=1) ** 2
    mu_sum = sum(mu_samples)

    mu = (norm_1_sq * mu_sum) / count
    return mu


def ground_state_constants(spectral_gap, eeta, eps, prob, obs_norm):
    """
    Calculating the parameters for the LCU based on the given constants.

    Values calculated:
        - t: Base time for evolution
        - count: Number of iterations (T in the paper)
        - gamma: Error gap between LCU and exact function
        - delta_t: Time factor (refer paper)
        - m: M from paper, the max value of j in the loop

    Calculations:
        - t = (1/delta^2) * log(||O|| / eeta * eps)

        - count = (||O||^2 ln(2/delta)) / (eps^2 * eeta^4)

        - gamma = eps * eeta^2 / 12 ||O||

        - delta_t = 1 / (sqrt(2t) + sqrt(2log(5/gamma)))
    """

    t = (1 / spectral_gap**2) * np.log2(1 / (eeta * eps))
    count = int(((obs_norm**2) * np.log(2 / prob)) / (eps**2 * eeta**4))

    gamma = (eps * (eeta**2)) / (12 * obs_norm)
    m = int(
        np.ceil(
            np.sqrt(2)
            * (np.sqrt(t) + np.sqrt(np.log2(5 / gamma)))
            * np.sqrt(np.log2(4 / gamma))
        )
    )
    delta_t = 1 / (np.sqrt(2 * t) + np.sqrt(2 * np.log2(5 / gamma)))
    return {"t": t, "count": count, "gamma": gamma, "m": m, "delta_t": delta_t}


def calculate_lcu_constants(m, delta_t, t):
    """
    Coefficients are calculated by:
        c_j = (delta_t / sqrt(2pi)) * e^{-j^2 delta_t^2 / 2}

    The overall time value for given j is calculated by:
        t_j = j * delta_t * sqrt(2*t)
    """
    # from -M to M
    js = np.arange(-m, m + 1)
    coeffs = (delta_t / np.sqrt(2 * np.pi)) * np.exp(-1 * js**2 * (delta_t**2) / 2)
    times = js * delta_t * np.sqrt(2 * t)

    return coeffs, times


class GroundState:
    # def calculate_magnetization(ham, observable, eeta, eps, prob, **kwargs):
    def __init__(self, synthesizer: Synthesizer, observable: Hamiltonian, **kwargs):
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
        self.synth = synthesizer
        obs_x = SparsePauliOp(["X"], [1.0])
        run_obs = obs_x.tensor(observable.sparse_repr)

        self.run_obs = run_obs.to_matrix()
        # TODO: Replace with `spectral_norm`
        self.obs_norm = 1

        self.params = {x: kwargs[x] for x in ["eeta", "eps", "prob"]}
        self.ground_params = ground_state_constants(
            self.synth.ham.spectral_gap,
            self.params["eeta"],
            self.params["eps"],
            self.params["prob"],
            self.obs_norm,
        )

        self.lcu_coeffs, self.lcu_times = calculate_lcu_constants(
            self.ground_params["m"],
            self.ground_params["delta_t"],
            self.ground_params["t"],
        )
        self.lcu_indices = list(range(len(self.lcu_times)))

        self.eye = np.identity(self.synth.ham.eig_vec.shape[0])

        # Normalized plus state of size N: [1...1] / sqrt{N}
        plus_state = np.ones_like(self.synth.ground_state) / np.sqrt(
            len(self.synth.ground_state)
        )
        # all zero states
        computational_state = np.zeros_like(self.synth.ground_state)
        computational_state[0] = 1

        init_state = close_state(
            state=self.synth.ground_state,
            overlap=self.params["eeta"],
            other_states=np.array([plus_state, computational_state]),
        )
        init_state = init_state.reshape(-1, 1)
        init_complete = np.kron(PLUS, init_state)

        npt.assert_almost_equal(np.sum(np.abs(init_complete) ** 2), 1)

        self.rho_init = np.outer(init_complete, init_complete.conj())

        npt.assert_almost_equal(np.trace(self.rho_init), 1)

    def get_unitary(self, ind: int):
        """
        Uses the synthesizer for constructing Hamiltonian simulation for given
        lcu time index.

        Inputs:
            - ind: Sampled index
        """
        return self.synth.get_unitary(self.lcu_times[ind])

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

    def post_v1(self, ind):
        final_rho = self.rho_init.copy()
        v1 = self.control_unitary(ind, control_val=1)
        final_rho = v1 @ final_rho @ v1.conj().T
        npt.assert_almost_equal(np.trace(final_rho), 1)
        return final_rho

    def post_v1v2(self, i1, i2):
        final_rho = self.post_v1(i1)

        v2 = self.control_unitary(i2, control_val=0)
        final_rho = v2 @ final_rho @ v2.conj().T

        npt.assert_almost_equal(np.trace(final_rho), 1)

        return final_rho

    def calculate_mu(self):
        p = np.abs(self.lcu_coeffs) / np.linalg.norm(self.lcu_coeffs, ord=1)
        count = self.ground_params["count"]

        print("Entering loop")
        results = []

        samples_count = Counter(
            [tuple(x) for x in np.random.choice(self.lcu_indices, p=p, size=(count, 2))]
        )

        total_count = 0

        for sample in sorted(samples_count.keys()):
            s_count = samples_count[sample]
            if total_count % 10 == 0:
                print(f"running: {total_count} out of {count}")
            total_count += s_count

            final_rho = self.post_v1v2(sample[0], sample[1])

            result = np.trace(np.abs(self.run_obs @ final_rho))
            results.append(result * s_count)

        assert total_count == count

        magn_h = calculate_mu(results, count, self.lcu_coeffs)
        return magn_h
