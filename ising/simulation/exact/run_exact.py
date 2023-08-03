import argparse
import json
import numpy as np

from ising.hamiltonian import parametrized_ising
from ising.observables import overall_magnetization
from ising.utils import read_input_file, close_state
from ising.simulation.exact import ExactSimulation


def run_exact(paras):
    qubit_wise_answers = {}
    times = np.linspace(0, paras["time"], paras["count_time"])
    h_values = np.linspace(
        10 ** paras["start_h"], 10 ** paras["end_h"], paras["count_h"]
    )

    for num_qubit in range(paras["start_qubit"], paras["end_qubit"] + 1):
        observable = overall_magnetization(num_qubit)

        h_wise_answers = {}
        for h in h_values:
            print(f"Running for {num_qubit} qubits and h:{h}")
            ham = parametrized_ising(num_qubit, h)
            circuit_manager = ExactSimulation(ham)
            ground_state = circuit_manager.ground_state
            init_state = close_state(ground_state, paras["overlap"])
            rho_init = np.outer(init_state, init_state.conj().T)

            ans = circuit_manager.get_observations(rho_init, observable.matrix, times)
            h_wise_answers[h] = ans

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

    file_name = f"data/output/magnetization_exact_{parameters['start_qubit']}_to_{parameters['end_qubit']}.json"

    print(f"Saving results at: {file_name}")
    with open(file_name, "w") as file:
        json.dump(results, file)


def test_main():
    parameters = read_input_file("data/input/test-ising-trotter.json")

    results = run_exact(parameters)

    file_name = f"data/output/magnetization_exact_{parameters['start_qubit']}_to_{parameters['end_qubit']}.json"

    print(f"Saving results at: {file_name}")
    with open(file_name, "w") as file:
        json.dump(results, file)


if __name__ == "__main__":
    main()
