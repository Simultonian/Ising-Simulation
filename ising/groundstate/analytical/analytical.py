from ising.hamiltonian import parametrized_ising
import numpy as np


def generate_basis(num_qubits):
    binary = []
    n = 2**num_qubits

    # Iterate from 1 to n
    for i in range(n):
        # Get binary representation of i as string
        binary_str = np.binary_repr(i)

        binary_str = binary_str.zfill(num_qubits)
        # Convert binary string to list of integers
        binary_list = list(map(int, binary_str))
        # Append list of integers to binary list
        binary.append(binary_list)

    return binary


def magnetization(state, basis):
    m = 0
    for i, bstate in enumerate(basis):
        b_m = 0
        for spin in bstate:
            if spin:
                b_m += state[i] ** 2
            else:
                b_m -= state[i] ** 2
        b_m /= len(bstate)
        assert b_m <= 1
        m += abs(b_m)
    return m


def get_qubit_magnetization(num_qubit, h_vals):
    basis = generate_basis(num_qubit)

    # Get Ising Hamiltonian
    h_magn = {}
    for h in h_vals:
        print(f"Running for {num_qubit} qubits and h={h}")
        ham = parametrized_ising(num_qubit, -1 * h, -1, normalize=False)
        eigenval, eigenvec = np.linalg.eig(ham.matrix)

        # Get the ground state of the m
        min_ind = np.argmin(eigenval)
        ground_state = eigenvec[:, min_ind]
        h_magn[h] = magnetization(ground_state, basis)

    return h_magn


def ising_magnetization_analytical(h_low, h_high, h_count, start_q, end_q, count_q):
    all_magn = {}
    h_vals = [10**x for x in np.linspace(h_low, h_high, h_count)]
    for num_qubits in range(start_q, end_q + 1, count_q):
        all_magn[num_qubits] = get_qubit_magnetization(num_qubits, h_vals)

    return all_magn
