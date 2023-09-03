import argparse
import json
import numpy as np

from ising.hamiltonian import parametrized_ising
from ising.utils import read_input_file


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


def run_exact(paras):
    qubit_wise_answers = {}
    h_values = np.linspace(
        10 ** paras["start_h"], 10 ** paras["end_h"], paras["count_h"]
    )

    for num_qubit in range(paras["start_qubit"], paras["end_qubit"] + 1):
        # Generating basis for given number of qubits
        print("Generating basis")
        basis = generate_basis(num_qubit)

        h_wise_answers = {}
        for h in h_values:
            print(f"Running for {num_qubit} qubits and h:{h}")
            ham = parametrized_ising(num_qubit, h)
            ground_state = ham.ground_state

            magn_h = magnetization(ground_state, basis)

            h_wise_answers[h] = magn_h

        qubit_wise_answers[num_qubit] = h_wise_answers

    return qubit_wise_answers


def main():
    parser = argparse.ArgumentParser(
        description="Overall magnetization of Ising using Qiskit"
    )
    parser.add_argument("--input", type=str, help="File for input parameters.")
    args = parser.parse_args()

    parameters = read_input_file(args.input)

    results = run_exact(parameters)

    file_name = f"data/singlelcu/output/magnetization_exact_{parameters['start_qubit']}_to_{parameters['end_qubit']}.json"

    print(f"Saving results at: {file_name}")
    with open(file_name, "w") as file:
        json.dump(results, file)


def test_main():
    parameters = read_input_file("data/singlelcu/input/default.json")

    results = run_exact(parameters)

    file_name = f"data/singlelcu/output/magnetization_exact_{parameters['start_qubit']}_to_{parameters['end_qubit']}.json"

    print(f"Saving results at: {file_name}")
    with open(file_name, "w") as file:
        json.dump(results, file)


if __name__ == "__main__":
    main()
