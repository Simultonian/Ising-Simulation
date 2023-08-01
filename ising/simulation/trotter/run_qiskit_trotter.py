import argparse
import json
import numpy as np
from typing import Callable

from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp

from ising.simulation.exact import ExactSimulation
from ising.hamiltonian import parametrized_ising, TROTTER_REP_FUNC
from ising.observables import overall_magnetization
from ising.utils import read_input_file, close_state
from ising.simulation.trotter import QiskitTrotter, TROTTER_MAP


def run_qiskit_trotter(
    paras, rep_counter: Callable[[SparsePauliOp, float, float], int], synthesis
):
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
            assigned_ham = ham.assign_parameters({h_para: h})
            assert assigned_ham is not None

            exact_simulator = ExactSimulation(assigned_ham, observable)
            ground_state = exact_simulator.ground_state.reshape(-1, 1)

            reps = rep_counter(assigned_ham, paras["time"], paras["error"])
            print(f"Running Trotter for h:{h} qubits:{num_qubit} reps:{reps}")

            synth = synthesis(reps)
            simulator = QiskitTrotter(
                assigned_ham, exact_simulator.observable, synthesis=synth
            )
            rho_init = close_state(
                np.outer(ground_state, ground_state.conj()), paras["overlap"]
            )

            ans = simulator.get_observations(rho_init, para_times=times)
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

    rep_counter = TROTTER_REP_FUNC[parameters["method"]]
    Synthesis = TROTTER_MAP[parameters["method"]]
    synthesis = Synthesis

    results = run_qiskit_trotter(parameters, rep_counter, synthesis)

    file_name = f"data/output/magnetization_{parameters['method']}_{parameters['start_qubit']}_to_{parameters['end_qubit']}.json"

    print(f"Saving results at: {file_name}")
    with open(file_name, "w") as file:
        json.dump(results, file)


if __name__ == "__main__":
    main()
