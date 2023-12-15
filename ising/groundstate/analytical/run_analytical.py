import argparse
import json
import numpy as np

from ising.utils import read_input_file
from ising.groundstate.analytical import get_qubit_magnetization


def run_analytical(paras):
    qubit_wise_answers = {}
    h_values = np.linspace(
        10 ** paras["start_h"], 10 ** paras["end_h"], paras["count_h"]
    )

    qubits = map(
        int, np.linspace(paras["start_qubit"], paras["end_qubit"], paras["qubit_count"])
    )

    for num_qubit in qubits:
        qubit_wise_answers[num_qubit] = get_qubit_magnetization(num_qubit, h_values)

    return qubit_wise_answers


def file_analytical(file_name):
    parameters = read_input_file(file_name)
    results = run_analytical(parameters)

    file_name = f"data/groundstate/analytical_{parameters['start_qubit']}_to_{parameters['end_qubit']}.json"

    print(f"Saving results at: {file_name}")
    with open(file_name, "w") as file:
        json.dump(results, file)


def main():
    parser = argparse.ArgumentParser(
        description="Overall magnetization of Ising using Qiskit"
    )
    parser.add_argument("--input", type=str, help="File for input parameters.")
    args = parser.parse_args()
    file_analytical(args.input)


def test_main():
    file_analytical("data/input/groundstate.json")


if __name__ == "__main__":
    np.random.seed(42)
    main()
