import numpy as np
from ising.hamiltonian import parametrized_ising
from scipy.sparse import linalg
class TestIsing:
    def test_ising(self):
        h = 0.1
        for qubits in range(2, 10, 2):
            ham = parametrized_ising(qubits, h, normalize=False)
            mat = ham.sparse_repr.to_matrix(sparse=True)
            eigval, _ = linalg.eigs(mat, k=2, which='SR')
            spectral_gap = abs(eigval[1] - eigval[0])
            np.testing.assert_allclose(spectral_gap, ham.spectral_gap, rtol=1e-5)
