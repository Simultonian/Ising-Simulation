from qiskit.quantum_info import Operator, SparsePauliOp, Pauli
from qiskit.opflow.converters import PauliBasisChange
import numpy as np

# Define the Pauli matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Define the unitary matrix (example: Pauli-X gate)
unitary_matrix = SparsePauliOp(["XI", "ZZ", "YZ"], [1, 2, 0.5]).to_matrix()

# Create a list of Pauli matrices and their labels
pauli_matrices = [I, X, Y, Z]
pauli_labels = ["I", "X", "Y", "Z"]

# Function to convert unitary matrix to Pauli decomposition
def unitary_to_pauli_decomposition(unitary):
    from itertools import product

    qubit_count = int(np.log2(unitary.shape[0]))
    pauli_strings, coefficients = [], []
    for inds in product(range(len(pauli_labels)), repeat=qubit_count):
        label, pauli = [], None
        for ind in inds:
            label.append(pauli_labels[ind])
            if pauli is None:
                pauli = pauli_matrices[ind]
            else:
                pauli = np.kron(pauli, pauli_matrices[ind])

        assert pauli is not None
        coeff = np.trace(np.dot(pauli, unitary)) / (2 ** qubit_count)

        if np.abs(coeff) > 1e-10:  # Filter out small coefficients
            pauli_strings.append("".join(label))
            coefficients.append(coeff)

    return pauli_strings, coefficients

# Get the Pauli decomposition
paulis, coeffs = unitary_to_pauli_decomposition(unitary_matrix)

# Print the result
for coeff, label in zip(coeffs, paulis):
    print(f"{np.real(coeff):.4f} * {label}")
