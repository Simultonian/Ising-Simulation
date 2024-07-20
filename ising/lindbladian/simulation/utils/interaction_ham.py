import numpy as np
import json
from qiskit.quantum_info import SparsePauliOp
from ising.utils import unitary_to_pauli_decomposition
SIGMA_MINUS = np.array([[0, 1], [0, 0]])
SIGMA_PLUS = np.array([[0, 0], [1, 0]])


def interaction_hamiltonian(qubit_count):
    """
    Construct a `2*qubit_count` Hamiltonian for each interaction point.
    There will be `qubit_count` of them, acting on two qubits each

    Input:
        - qubit_count: Size of the chain
        - gamma: Strength of the Hamiltonian
    """
    ham_ints = []
    for _site in range(qubit_count):
        sys_site, env_site = _site, _site + qubit_count

        ham_int1, ham_int2 = None, None
        for pos in range(2*qubit_count):
            cur_op1, cur_op2 = None, None
            if pos == sys_site:
                cur_op1, cur_op2 = SIGMA_PLUS, SIGMA_MINUS
            elif pos == env_site:
                cur_op1, cur_op2 = SIGMA_MINUS, SIGMA_PLUS
            else:
                cur_op1, cur_op2 = np.eye(2), np.eye(2)

            if ham_int1 is None or ham_int2 is None:
                ham_int1, ham_int2 = cur_op1, cur_op2
            else:
                ham_int1 = np.kron(ham_int1, cur_op1)
                ham_int2 = np.kron(ham_int2, cur_op2)

        assert ham_int1 is not None and ham_int2 is not None

        ham_ints.append(ham_int1 + ham_int2)

    return ham_ints


def save_interaction_hams(qubit_count):
    ham_ints = interaction_hamiltonian(qubit_count)
    result = {}
    for ind, ham_int in enumerate(ham_ints):
        paulis, coeffs =  unitary_to_pauli_decomposition(ham_int)
        result[ind] = {"paulis": paulis, "coeffs": coeffs}

    file_name = f"data/lindbladian/hamiltonian/size_{qubit_count}.json"
    with open(file_name, "w") as file:
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
