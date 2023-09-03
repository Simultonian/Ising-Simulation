import numpy as np
from numpy.typing import NDArray

from qiskit.quantum_info import Pauli, Operator, SparsePauliOp
from qiskit.circuit import Parameter

from ising.utils import simdiag
from ising.simulation.trotter import Lie, LieCircuit, GroupedLieCircuit, GroupedLie
from ising.hamiltonian import general_grouping, parametrized_ising


def pauli_matrix(eig_val:NDArray, eig_vec:NDArray, eig_inv:NDArray, time: float) -> NDArray:
    return eig_vec @ np.diag(np.exp(complex(0, -1) * time * eig_val)) @ eig_inv

def test_grouping():
    ising = [Pauli("XI"), Pauli("IX"), Pauli("ZZ")]
    group_mapping = general_grouping(ising)
    exp_mapping= [[Pauli("ZZ")], [Pauli("XI"), Pauli("IX")]]
    assert group_mapping == exp_mapping


def test_exp():
    t = 1.0
    ising = [Pauli("ZZ"), Pauli("XI"), Pauli("IX")]
    groups = general_grouping(ising)

    def group_exp(group):
        ms = [np.array(x.to_matrix()) for x in group]
        final_op = np.eye(2**2)
        for m in ms:
            ea, ee = np.linalg.eig(m)
            ei = np.linalg.inv(ee)
            op = pauli_matrix(ea, ee, ei, t)
            final_op = np.dot(final_op, op)

        return final_op

    def fast_exp(group):
        ms = [np.array(x.to_matrix()) for x in group]
        e_vals, e_vec = simdiag(ms)
        e_inv = np.linalg.inv(e_vec)

        e_sum = np.sum(e_vals, axis=0)
        return e_vec @ np.diag(np.exp(complex(0, -1) * t * e_sum)) @ e_inv

    g_final = np.eye(2**2)
    f_final = np.eye(2**2)

    for group in groups:
        g_final = np.dot(group_exp(group), g_final)
        f_final = np.dot(fast_exp(group), f_final)

    norm_final = group_exp(list(reversed(ising)))

    np.testing.assert_almost_equal(g_final, f_final)
    np.testing.assert_almost_equal(g_final, norm_final)

def test_circuit_match():
    time = 2.0
    p, coeff = (Pauli("ZZ"), -1)
    m = p.to_matrix()
    ea, ee = np.linalg.eig(m)
    ei = np.linalg.inv(ee)

    ea *= coeff

    op = ee @ np.diag(np.exp(complex(0, -1) * time * ea)) @ ei 

    lie = Lie()
    circuit = lie.atomic_evolution(p, coeff * time)
    r_op = np.array(Operator.from_circuit(circuit).reverse_qargs().data).astype(
        np.complex128
    )

    np.testing.assert_almost_equal(r_op, op)


def test_lie_grouped():
    time = 1.0
    op = SparsePauliOp([Pauli("XI"), Pauli("IX"), Pauli("ZZ")], [2, 2, -1])

    group_map = {Pauli("XI"): (0, 0), Pauli("IX"): (1, 0), Pauli("ZZ"): (0, 1)}

    g_lie = GroupedLie()
    lie = Lie()
    p_map = lie.parameterized_map(op, time)
    eig_list = g_lie.svd_map([[Pauli("XI"), Pauli("IX")], [Pauli("ZZ")]])

    for p, coeff in zip(op.paulis.label_iter(), op.coeffs):
        p_ind, g_ind = group_map[Pauli(p)]
        eig_list[g_ind][0][p_ind] = eig_list[g_ind][0][p_ind] * coeff

    def check_circuit_match(p):
        p_ind, g_ind = group_map[p]
        circuit = p_map[p]

        # No reverse_qargs makes the correct
        r_op = np.array(Operator.from_circuit(circuit).data).astype(
            np.complex128
        )


        eas, ee, ei = eig_list[g_ind]
        ea = eas[p_ind]

        e_op = ee @ np.diag(np.exp(complex(0, -1) * time * ea)) @ ei
        np.testing.assert_allclose(r_op, e_op, rtol=1e-7, atol=1e-7)
    
    for p in op.paulis:
        check_circuit_match(p)


def test_pauli_lie():
    h_para = Parameter("h")
    h_value = 1.0
    error = 0.1
    time = 1.0

    parametrized_ham = parametrized_ising(2, h_para)

    def get_pauli(circuit_synthesis, pauli, time):
        circuit_manager = circuit_synthesis(parametrized_ham, h_para, error)
        circuit_manager.subsitute_h(h_value)
        circuit_manager.construct_parametrized_circuit()

        return circuit_manager.pauli_matrix(pauli, time, 1)


    for pauli in parametrized_ham.sparse_repr.paulis:
        lie_matrix = get_pauli(LieCircuit, pauli, time)
        grouped_matrix = get_pauli(GroupedLieCircuit, pauli, time)
        np.testing.assert_almost_equal(lie_matrix, grouped_matrix)

def test_grouped_pauli():
    h_para = Parameter("h")
    error = 0.1

    parametrized_ham = parametrized_ising(2, h_para)

    lie_circuit = LieCircuit(parametrized_ham, h_para, error)
    grouped_circuit = GroupedLieCircuit(parametrized_ham, h_para, error)

    for h_value in [1.0, 0.5, 2.0]:
        lie_circuit.subsitute_h(h_value)
        grouped_circuit.subsitute_h(h_value)
        lie_circuit.construct_parametrized_circuit()
        grouped_circuit.construct_parametrized_circuit()

        for time in [1.0, 10.0, 5.0]:
            for pauli in parametrized_ham.sparse_repr.paulis:
                def get_pauli(circuit_manager, time):
                    return circuit_manager.pauli_matrix(pauli, time, 1)

                lie_matrix = get_pauli(lie_circuit, time)
                grouped_matrix = get_pauli(grouped_circuit, time)
                np.testing.assert_almost_equal(lie_matrix, grouped_matrix)

def test_grouped_matrix():
    h_para = Parameter("h")
    error = 0.1

    parametrized_ham = parametrized_ising(2, h_para)

    lie_circuit = LieCircuit(parametrized_ham, h_para, error)
    grouped_circuit = GroupedLieCircuit(parametrized_ham, h_para, error)

    for h_value in [1.0, 0.9, 2.0]:
        lie_circuit.subsitute_h(h_value)
        grouped_circuit.subsitute_h(h_value)
        lie_circuit.construct_parametrized_circuit()
        grouped_circuit.construct_parametrized_circuit()

        for time in [1.0, 10.0, 5.0]:
            def get_pauli(circuit_manager, time):
                return circuit_manager.matrix(time)

            lie_matrix = get_pauli(lie_circuit, time)
            grouped_matrix = get_pauli(grouped_circuit, time)
            np.testing.assert_almost_equal(lie_matrix, grouped_matrix)
