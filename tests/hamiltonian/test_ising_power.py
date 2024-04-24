from ising.hamiltonian import parametrized_ising_power
from qiskit.circuit import Parameter


class TestIsing:
    def test_ising(self):
        h_value = 1.0
        ham = parametrized_ising_power(1, h_value)
        assert ham.sparse_repr.paulis == ["X"]

    def test_long_range_two(self):
        h = 1.0
        ham = parametrized_ising_power(2, h=h, power=1)
        assert ham.sparse_repr.paulis == ['ZZ', 'XI', 'IX']
        assert all(ham.sparse_repr.coeffs == [-1.+0.j,  1.+0.j,  1.+0.j])

    def test_long_range_three(self):
        h = 1.0
        ham = parametrized_ising_power(3, h=h, power=1)
        assert ham.sparse_repr.paulis == ['ZZI', 'ZIZ', 'IZZ', 'XII', 'IXI', 'IIX']
        assert all(ham.sparse_repr.coeffs == [-1.+0.j,  -1/2, -1, 1.+0.j,  1.+0.j, 1])
