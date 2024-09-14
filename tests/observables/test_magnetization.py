from ising.observables import overall_magnetization
import numpy as np


ZERO = np.array([[1], [0]])

def _random_psi(qubit_count):
    real_psi = np.random.uniform(-1, 1, 2**qubit_count)
    norm = sum([np.abs(x) for x in real_psi]) ** 0.5
    return real_psi / norm

class TestMagnetization:
    def test_magnetization(self):
        obs = overall_magnetization(2)

        coeffs = np.array([1 / 2] * 2)
        assert all(obs.sparse_repr.coeffs == coeffs)


def test_overall_zero():
    obs = overall_magnetization(num_qubits=4)

    psi = ZERO.copy()
    for _ in range(3):
        psi = np.kron(psi, ZERO)

    rho = np.outer(psi, psi.conj().T)
    magn  = np.trace(obs.matrix @ rho)
    assert magn == 1

def test_random_psi():
    magns = []
    obs = overall_magnetization(num_qubits=4)

    for _ in range(100):
        psi = _random_psi(4)
        rho = np.outer(psi, psi.conj().T)
        magn = np.trace(np.abs(obs.matrix @ rho))

        magns.append(magn)

    assert False
