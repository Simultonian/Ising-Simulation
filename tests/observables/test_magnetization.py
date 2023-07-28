from ising.observables import overall_magnetization 
import numpy as np


class TestMagnetization:
    def test_magnetization(self):
        obs = overall_magnetization(2)

        coeffs = np.array([1/2] * 2)
        assert all(obs.coeffs == coeffs)
