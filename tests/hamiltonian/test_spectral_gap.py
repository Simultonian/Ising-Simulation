from qiskit.quantum_info import SparsePauliOp, Pauli
import numpy as np


class TestSpectralGap:
    def test_spectral_simple(self):
        terms = ["Z"]
        coeffs = [1.0]
        ham = SparsePauliOp(terms, coeffs=coeffs)
        mat = ham.to_matrix()

        eig_val, eig_vec = np.linalg.eig(mat)
