import numpy as np
import json
from qiskit.quantum_info import SparsePauliOp
from ising.utils import unitary_to_pauli_decomposition
from ising.lindbladian.simulation.multi_cm_damping import interaction_hamiltonian
SIGMA_MINUS = np.array([[0, 1], [0, 0]])
SIGMA_PLUS = np.array([[0, 0], [1, 0]])

SIGMA_PLUS_SPARSE = SparsePauliOp(["X", "Y"], [0.5, -0.5j])
SIGMA_MINUS_SPARSE = SparsePauliOp(["X", "Y"], [0.5, 0.5j])
EYE = SparsePauliOp(["I"], [1])


def interaction_hamiltonian_sparse(QUBIT_COUNT, gamma=1):
    """
    Construct a `QUBIT_COUNT+1` Hamiltonian for each interaction point.
    There will be `QUBIT_COUNT` of them, acting on two qubits each

    Input:
        - QUBIT_COUNT: Size of the chain
        - gamma: Strength of the Hamiltonian
    """
    ham_ints = []
    for _site in range(QUBIT_COUNT):
        sys_site, env_site = _site, QUBIT_COUNT + 1

        ham_int1, ham_int2 = None, None
        for pos in range(QUBIT_COUNT + 1):
            cur_op1, cur_op2 = None, None
            if pos == sys_site:
                cur_op1, cur_op2 = SIGMA_PLUS_SPARSE, SIGMA_MINUS_SPARSE
            elif pos == env_site:
                cur_op1, cur_op2 = SIGMA_MINUS_SPARSE, SIGMA_PLUS_SPARSE
            else:
                cur_op1, cur_op2 = EYE, EYE

            if ham_int1 is None or ham_int2 is None:
                ham_int1, ham_int2 = cur_op1, cur_op2
            else:
                ham_int1 = ham_int1.tensor(cur_op1)
                ham_int2 = ham_int2.tensor(cur_op2)

        assert ham_int1 is not None and ham_int2 is not None

        ham_ints.append(np.sqrt(gamma) * (ham_int1 + ham_int2))

    return ham_ints


def save_interaction_hams(qubit_count):
    ham_ints_sparse = interaction_hamiltonian_sparse(qubit_count)
    result = {}
    for ind, ham_int in enumerate(ham_ints_sparse):
        result[ind] = {"paulis": [str(x) for x in ham_int.paulis], "coeffs": [x.real for x in ham_int.coeffs]}

    file_name = f"data/lindbladian/hamiltonian/size_{qubit_count}.json"
    with open(file_name, "w+") as file:
        json.dump(result, file)
        
    print(f"Saved interaction_hams for {qubit_count} qubits at: {file_name}")

def load_interaction_hams(qubit_count):
    with open(f"data/lindbladian/hamiltonian/size_{qubit_count}.json", "r") as file:
        result = json.load(file)

    ham_ints = []
    for _, desc in result.items():
        ham_ints.append(SparsePauliOp(desc["paulis"], desc["coeffs"]))

    return ham_ints 

if __name__ == "__main__":
    save_interaction_hams(3)
