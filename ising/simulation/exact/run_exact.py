import argparse
import json
import numpy as np
from ising.simulation.exact import ExactSimulation
from qiskit.circuit import Parameter
from ising.hamiltonian import parametrized_ising
from ising.observables import overall_magnetization
from ising.utils import read_input_file, close_state


def run_exact(paras):
    qubit_wise_answers = {}

    times = np.linspace(0, paras["time"], paras["count_time"])

    for num_qubit in range(paras["start_qubit"], paras["end_qubit"] + 1):
        observable = overall_magnetization(num_qubit).to_matrix()
        h_para = Parameter("h")
        ham = parametrized_ising(num_qubit, h_para)

        h_wise_answers = {}

        h_values = np.linspace(
            10 ** paras["start_h"], 10 ** paras["end_h"], paras["count_h"]
        )
        for h in h_values:
            print(f"Running for {num_qubit} qubits and h:{h}")
            assigned_ham = ham.assign_parameters({h_para: h})
            assert assigned_ham is not None

            simulator = ExactSimulation(assigned_ham, observable)
            ground_state = simulator.ground_state.reshape(-1, 1)
            rho_init = close_state(
                np.outer(ground_state, ground_state.conj()), paras["overlap"]
            )

            ans = simulator.get_observations(rho_init, para_times=times)
            h_wise_answers[h] = ans

        qubit_wise_answers[num_qubit] = h_wise_answers

    return qubit_wise_answers


def main():
    parser = argparse.ArgumentParser(description="Overall magnetization of Ising")
    parser.add_argument("--input", type=str, help="File for input parameters.")
    args = parser.parse_args()

    parameters = read_input_file(args.input)
    results = run_exact(parameters)

    file_name = f"data/output/magnetization_exact_{parameters['start_qubit']}_to_{parameters['end_qubit']}.json"

    print(f"Saving results at: {file_name}")
    with open(file_name, "w") as file:
        json.dump(results, file)


if __name__ == "__main__":
    main()
