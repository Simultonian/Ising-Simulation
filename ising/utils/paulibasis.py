import numpy as np

I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

PAULI_MATRICES = [I, X, Y, Z]
PAULI_LABELS = ["I", "X", "Y", "Z"]

def unitary_to_pauli_decomposition(unitary):
    from itertools import product

    qubit_count = int(np.log2(unitary.shape[0]))
    pauli_strings, coefficients = [], []
    for inds in product(range(len(PAULI_LABELS)), repeat=qubit_count):
        label, pauli = [], None
        for ind in inds:
            label.append(PAULI_LABELS[ind])
            if pauli is None:
                pauli = PAULI_MATRICES[ind]
            else:
                pauli = np.kron(pauli, PAULI_MATRICES[ind])

        assert pauli is not None
        coeff = np.trace(np.dot(pauli, unitary)) / (2 ** qubit_count)
        assert coeff.imag < 1e-10
        coeff = coeff.real

        if np.abs(coeff) > 1e-10:  # Filter out small coefficients
            pauli_strings.append("".join(label))
            coefficients.append(coeff)

    return pauli_strings, coefficients
