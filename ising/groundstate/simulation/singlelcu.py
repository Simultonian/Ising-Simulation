from functools import lru_cache
import numpy as np
import numpy.testing as npt
from collections import Counter

from qiskit.quantum_info import SparsePauliOp

from ising.hamiltonian import Hamiltonian
from ising.utils.constants import ONEONE, ZEROZERO, PLUS
from ising.utils import close_state, MAXSIZE
from ising.groundstate.simulation.utils import (
    spectral_norm,
    calculate_mu,
    ground_state_constants,
    calculate_lcu_constants,
)


class LCUSynthesizer:
    def __init__(self, synthesizer, observable: Hamiltonian, **kwargs):
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
            self.synth.ham_subbed.spectral_gap,
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
        return self.synth.matrix(self.lcu_times[ind])

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
            if total_count % 100 == 0:
                print(f"running: {total_count} out of {count}")
            total_count += s_count

            final_rho = self.post_v1v2(sample[0], sample[1])

            result = np.trace(np.abs(self.run_obs @ final_rho))
            results.append(result * s_count)

        assert total_count == count

        magn_h = calculate_mu(results, count, self.lcu_coeffs)
        return magn_h
