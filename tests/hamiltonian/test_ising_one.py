from ising.hamiltonian import parametrized_ising, substitute_parameter
from qiskit.circuit import Parameter


class TestIsing:
    def test_ising(self):
        h = Parameter("h")
        h_value = 1.0
        ham = parametrized_ising(1, h)

        ham_1 = substitute_parameter(ham, h, h_value)

        assert ham_1 is not None
        assert ham_1.sparse_repr.paulis == ["X"]
