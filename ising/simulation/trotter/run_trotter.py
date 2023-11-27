import argparse
import json
import numpy as np

from qiskit.quantum_info import Operator
from qiskit.circuit import Parameter

from ising.hamiltonian import parametrized_ising
from ising.observables import overall_magnetization
from ising.utils import read_input_file, close_state
from ising.simulation.trotter import (
    LieCircuit,
    QDriftCircuit,
    SparseLie,
    GroupedLieCircuit,
    GroupedQDriftCircuit,
    TwoQDriftCircuit,
    GSQDriftCircuit,
)


def run_trotter(paras):
    qubit_wise_answers = {}
    times = np.linspace(
        paras["time"] / paras["count_time"], paras["time"], paras["count_time"]
    )
    h_values = np.linspace(
        10 ** paras["start_h"], 10 ** paras["end_h"], paras["count_h"]
    )

    method = paras["method"]

    if method == "lie":
        circuit_synthesis = LieCircuit
    elif method == "qdrift":
        circuit_synthesis = QDriftCircuit
    elif method == "sparse_lie":
        circuit_synthesis = SparseLie
    elif method == "grouped_lie":
        circuit_synthesis = GroupedLieCircuit
    elif method == "grouped_qdrift":
        circuit_synthesis = GroupedQDriftCircuit
    elif method == "two_qdrift_circuit":
        circuit_synthesis = TwoQDriftCircuit
    elif method == "gs_qdrift":
        circuit_synthesis = GSQDriftCircuit
    else:
        raise ValueError("Incorrect method:", method)

    qubits = np.linspace(paras["start_qubit"], paras["end_qubit"], paras["qubit_count"])

    for _num_qubit in qubits:
        num_qubit = int(_num_qubit)
        observable = overall_magnetization(num_qubit)
        h_para = Parameter("h")
        parametrized_ham = parametrized_ising(num_qubit, h_para)
        circuit_manager = circuit_synthesis(parametrized_ham, h_para, paras["error"])

        h_wise_answers = {}
        for h in h_values:
            print(f"Running for {num_qubit} qubits and h:{h}")
            circuit_manager.subsitute_h(h)
            circuit_manager.construct_parametrized_circuit()
            ground_state = circuit_manager.ham_subbed.ground_state
            init_state = close_state(ground_state, paras["overlap"])
            rho_init = np.outer(init_state, init_state.conj().T)

            rho_init = np.array(Operator(rho_init).reverse_qargs().data)

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

    results = run_trotter(parameters)

    file_name = f"data/simulation/magnetization_{parameters['method']}_{parameters['start_qubit']}_to_{parameters['end_qubit']}.json"

    print(f"Saving results at: {file_name}")
    with open(file_name, "w") as file:
        json.dump(results, file)


def test_main():
    parameters = read_input_file("data/input/default.json")

    results = run_trotter(parameters)

    file_name = f"data/simulation/magnetization_{parameters['method']}_{parameters['start_qubit']}_to_{parameters['end_qubit']}.json"

    print(f"Saving results at: {file_name}")
    with open(file_name, "w") as file:
        json.dump(results, file)


if __name__ == "__main__":
    np.random.seed(42)
    main()
