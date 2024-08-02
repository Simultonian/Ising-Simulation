import numpy as np
import json
from qiskit.quantum_info import SparsePauliOp
from ising.utils import unitary_to_pauli_decomposition
from ising.lindbladian.simulation.multi_cm_damping import interaction_hamiltonian
SIGMA_MINUS = np.array([[0, 1], [0, 0]])
SIGMA_PLUS = np.array([[0, 0], [1, 0]])


def save_interaction_hams(qubit_count):
    ham_ints = interaction_hamiltonian(qubit_count)
    result = {}
    for ind, ham_int in enumerate(ham_ints):
        paulis, coeffs =  unitary_to_pauli_decomposition(ham_int)
        result[ind] = {"paulis": paulis, "coeffs": coeffs}

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
