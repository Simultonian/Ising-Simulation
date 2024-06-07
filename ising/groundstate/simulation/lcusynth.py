from functools import lru_cache
import numpy as np
import numpy.testing as npt
from collections import Counter

from qiskit.quantum_info import SparsePauliOp

from ising.hamiltonian import Hamiltonian
from ising.utils.constants import PLUS
from ising.utils import close_state, MAXSIZE
from ising.groundstate.simulation.utils import (
    calculate_mu,
    ground_state_constants,
    calculate_lcu_constants,
)

from tqdm import tqdm


class LCUSynthesizer:
    def __init__(self, synthesizer, observable: Hamiltonian, **kwargs):
        """
        Code for running single ancilla LCU simulation using given synthesizer.

        Inputs:
            - synthesizer: Provides access for construction of unitaries.
            - observable: Observable to run LCU simulation on.

        Kwargs must contain:
            - overlap
            - error
            - success
        """
        self.synth = synthesizer

        obs_x = SparsePauliOp(["X"], [1.0])
        run_obs = obs_x.tensor(observable.sparse_repr)

        self.run_obs = run_obs.to_matrix()

        # TODO: Replace with `spectral_norm`
        self.obs_norm = 1

        # Useful constants for single-ancilla LCU groundstate
        self.overlap = kwargs["overlap"]
        self.error = kwargs["error"]
        self.success = kwargs["success"]

        self.ground_params = ground_state_constants(
            self.synth.ham_subbed.spectral_gap,
            self.overlap,
            self.error,
            self.success,
            self.obs_norm,
        )

        # Decomposition information fully contained in this
        self.lcu_coeffs, self.lcu_times = calculate_lcu_constants(
            self.ground_params["m"],
            self.ground_params["delta_t"],
            self.ground_params["t"],
        )

        """
        Preparation of Initial State
        """
        self.eye = np.identity(self.synth.ham_subbed.eig_vec.shape[0])

        # Normalized plus state of size N: [1...1] / sqrt{N}
        plus_state = np.ones_like(self.synth.ground_state) / np.sqrt(
            len(self.synth.ground_state)
        )
        # all zero states
        computational_state = np.zeros_like(self.synth.ground_state)
        computational_state[0] = 1

        init_state = close_state(
            state=self.synth.ground_state,
            overlap=self.overlap,
            other_states=np.array([plus_state, computational_state]),
        )
        init_state = init_state.reshape(-1, 1)
        init_complete = np.kron(PLUS, init_state)

        npt.assert_almost_equal(np.sum(np.abs(init_complete) ** 2), 1)
        self.psi_init = init_complete

    @lru_cache(maxsize=MAXSIZE)
    def post_v1(self, ind):
        psi_final = self.psi_init.copy()
        psi_final = self.synth.control_evolve(
            psi_final, self.lcu_times[ind], control_val=1
        )

        # npt.assert_allclose(np.sum(np.abs(psi_final) ** 2), 1, atol=1e-5)

        return psi_final

    def post_v1v2(self, i1, i2):
        psi_final = self.post_v1(i1)
        psi_final = self.synth.control_evolve(
            psi_final, self.lcu_times[i2], control_val=0
        )
        # npt.assert_allclose(np.sum(np.abs(psi_final) ** 2), 1, atol=1e-5)

        return psi_final

    def calculate_mu(self):
        p = np.abs(self.lcu_coeffs) / np.linalg.norm(self.lcu_coeffs, ord=1)
        count = self.ground_params["count"]

        print("Entering loop")
        results = []

        samples_count = Counter(
            [
                tuple(x)
                for x in np.random.choice(len(self.lcu_times), p=p, size=(count, 2))
            ]
        )

        total_count = 0

        with tqdm(total=count) as pbar:
            for sample in sorted(samples_count.keys()):
                s_count = samples_count[sample]
                total_count += s_count
                pbar.update(s_count)

                psi_final = self.post_v1v2(sample[0], sample[1])
                final_rho = np.outer(psi_final, psi_final.conj())

                result = np.trace(np.abs(self.run_obs @ final_rho))
                results.append(result * s_count)

            assert total_count == count

        return calculate_mu(results, count, self.lcu_coeffs)
