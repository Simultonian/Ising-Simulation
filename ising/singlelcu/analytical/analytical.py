from lcu.ham.ising import ising_1d
from lcu.ham.convert import to_pauli_op
import numpy as np
import argparse
import os


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


def get_all_magnetization(num_qubits, h_vals):
    # Generating basis for given number of qubits
    print("Generating basis")
    basis = generate_basis(num_qubits)

    # Get Ising Hamiltonian
    magn = []
    for h in h_vals:
        print(f"Running for {num_qubits} qubits and h={h}")
        ham = ising_1d(num_qubits, -1, -1 * h, normalize=False)
        ham_op = to_pauli_op(ham)
        ham_matrix = ham_op.to_matrix()
        assert ham_matrix is not None

        eigenval, eigenvec = np.linalg.eig(ham_matrix)

        # Get the ground state of the m
        min_ind = np.argmin(eigenval)
        ground_state = eigenvec[:, min_ind]
        magn_h = magnetization(ground_state, basis)
        magn.append(magn_h)

    return magn


def save(arr, start_q, end_q):
    # Saving after each run to avoid losing data after a crash
    file_name = f"data/exact/ising_exact_{start_q}_to_{end_q}.txt"

    print(f"Saving the data as a txt file: {file_name}")
    np.savetxt(file_name, np.array(arr))


def ising_magnetization_analytical(h_low, h_high, start_q, end_q):
    print(f"Running for qubit range: {start_q} to {end_q}")
    print(f"h value range: {h_low} to {h_high}")

    all_magn = []
    h_vals = [10**x for x in np.linspace(h_low, h_high, 10)]
    for num_qubits in range(start_q, end_q + 1):
        magn = get_all_magnetization(num_qubits, h_vals)
        all_magn.append(magn)
        save(all_magn, start_q, num_qubits)


def main():
    parser = argparse.ArgumentParser(description="Overall magnetization of Ising")
    parser.add_argument(
        "--h_low", type=float, default=-0.5, help='h lower range, default "-0.5"'
    )
    parser.add_argument(
        "--h_high", type=float, default=0.7, help='h upper range, default "0.7"'
    )
    parser.add_argument(
        "--start_q", type=int, default=3, help="qubit lower range, default 3"
    )
    parser.add_argument(
        "--end_q", type=int, default=5, help="qubit upper range, default 5"
    )
    args = parser.parse_args()

    ising_magnetization_analytical(args.h_low, args.h_high, args.start_q, args.end_q)


if __name__ == "__main__":
    main()
