import argparse
import json
import numpy as np

from qiskit.circuit import Parameter

from ising.hamiltonian import parametrized_ising
from ising.observables import overall_magnetization
from ising.utils import read_input_file, close_state
from ising.simulation.trotter import (
    QDriftCircuit,
    GroupedLieCircuit,
    GroupedQDriftCircuit,
    GSQDriftCircuit,
)


SYNTH_MAP = {
    "qdrift": QDriftCircuit,
    "grouped_lie": GroupedLieCircuit,
    "grouped_qdrift": GroupedQDriftCircuit,
    "gs_qdrift": GSQDriftCircuit,
}


def run_trotter(paras):
    qubit_wise_answers = {}
    times = np.linspace(
        paras["time"] / paras["count_time"], paras["time"], paras["count_time"]
    )
    h_values = np.linspace(
        10 ** paras["start_h"], 10 ** paras["end_h"], paras["count_h"]
    )

    method = paras["method"]

    if method not in SYNTH_MAP:
        raise ValueError("Incorrect method:", method)

    circuit_synthesis = SYNTH_MAP[method]

    qubits = map(
        int, np.linspace(paras["start_qubit"], paras["end_qubit"], paras["qubit_count"])
    )

    for num_qubit in qubits:
        h_para = Parameter("h")
        parametrized_ham = parametrized_ising(num_qubit, h_para)
        circuit_manager = circuit_synthesis(parametrized_ham, h_para, paras["error"])
        circuit_manager.substitute_obs(overall_magnetization(num_qubit))

        h_wise_answers = {}
        for h in h_values:
            print(f"Running for {num_qubit} qubits and h:{h}")
            circuit_manager.subsitute_h(h)
            circuit_manager.construct_parametrized_circuit()
            ground_state = circuit_manager.ground_state
            init_state = close_state(ground_state, paras["overlap"])

            ans = circuit_manager.get_observations(init_state, times)
            h_wise_answers[h] = ans

        qubit_wise_answers[num_qubit] = h_wise_answers

    return qubit_wise_answers


def file_trotter(file_name):
    parameters = read_input_file(file_name)

    results = run_trotter(parameters)

    file_name = f"data/simulation/magnetization_{parameters['method']}_{parameters['start_qubit']}_to_{parameters['end_qubit']}.json"

    print(f"Saving results at: {file_name}")
    with open(file_name, "w") as file:
        json.dump(results, file)


def main():
    parser = argparse.ArgumentParser(
        description="Overall magnetization of Ising using Qiskit"
    )
    parser.add_argument("--input", type=str, help="File for input parameters.")
    args = parser.parse_args()
    file_trotter(args.input)


def test_main():
    file_trotter("data/input/default.json")


if __name__ == "__main__":
    np.random.seed(42)
    main()
