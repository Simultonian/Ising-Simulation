from ising.hamiltonian import parametrized_ising
from qiskit.circuit import Parameter



class TestIsing:
    def test_ising(self):
        h = Parameter('h')
        h_value = 1.0
        ham = parametrized_ising(1, h)

        ham_1 = ham.assign_parameters({h: h_value})

        assert ham_1 is not None
        assert ham_1.paulis == ['X']
