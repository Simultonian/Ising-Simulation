import argparse
import json
import numpy as np

from qiskit.circuit import Parameter

from ising.hamiltonian import parametrized_ising
from ising.observables import overall_magnetization
from ising.utils import read_input_file, close_state
from ising.simulation.taylor import TaylorCircuit
from ising.simulation.taylor.taylor_sample import TaylorSample
from ising.simulation.taylor.taylor_single import TaylorSingle


def run_trotter(paras):
    qubit_wise_answers = {}
    times = np.linspace(
        paras["time"] / paras["count_time"], paras["time"], paras["count_time"]
    )
    h_values = np.linspace(
        10 ** paras["start_h"], 10 ** paras["end_h"], paras["count_h"]
    )

    method = paras["method"]

    if method == "taylor":
        circuit_synthesis = TaylorCircuit
    elif method == "taylor_sample":
        circuit_synthesis = TaylorSample
    elif method == "taylor_single":
        circuit_synthesis = TaylorSingle
    else:
        raise ValueError("This is Taylor file, method called:", method)

    for num_qubit in range(paras["start_qubit"], paras["end_qubit"] + 1):
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

            PLUS = np.array([[1], [1]]) / np.sqrt(2)
            init_complete = np.kron(PLUS, init_state)
            init_state = init_state.reshape(-1, 1)

            # Taking tensor product of the overlap state with `|+>` which is the state after `H`

            # Checking for norm
            np.testing.assert_almost_equal(np.sum(np.abs(init_complete) ** 2), 1)

            rho_init = np.outer(init_complete, init_complete.conj())

            ans = circuit_manager.get_observations(rho_init, observable, times)
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
    parameters = read_input_file("data/input/taylor.json")

    results = run_trotter(parameters)

    file_name = f"data/simulation/magnetization_{parameters['method']}_{parameters['start_qubit']}_to_{parameters['end_qubit']}.json"

    print(f"Saving results at: {file_name}")
    with open(file_name, "w") as file:
        json.dump(results, file)


if __name__ == "__main__":
    np.random.seed(42)
    main()
