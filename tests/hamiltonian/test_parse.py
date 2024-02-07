from ising.hamiltonian import parse
from qiskit.quantum_info import Pauli


class TestParse:
    def test_methane(self):
        name = "zee"
        ham = parse(name)
        terms, coeffs = (Pauli("Z"),), [1 + 0j]
        gap = 19.634
        assert all([p == t for p, t in zip(ham.paulis, terms)])
        assert all([p == t for p, t in zip(ham.coeffs, coeffs)])
        assert gap - ham.spectral_gap < 0.001
