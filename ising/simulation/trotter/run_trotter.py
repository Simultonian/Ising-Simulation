import argparse
import json
from numpy.typing import NDArray
import numpy as np

from qiskit.circuit import Parameter

from ising.hamiltonian import parametrized_ising, Hamiltonian
from ising.observables import overall_magnetization
from ising.utils import read_input_file, close_state
from ising.simulation.trotter import LieCircuit


def get_exact_unitary(ham: Hamiltonian, time: float) -> NDArray:
    return (
        ham.eig_vec
        @ np.diag(np.exp(complex(0, -1) * time * ham.eig_val))
        @ ham.eig_vec_inv
    )


def get_exact_observations(
    hamiltonian: Hamiltonian,
    observable: Hamiltonian,
    rho_init: NDArray,
    times: list[int],
) -> list[float]:
    results = []
    for time in times:
        unitary = get_exact_unitary(hamiltonian, time)
        rho_final = unitary @ rho_init @ unitary.conj().T
        result = np.trace(np.abs(observable.matrix @ rho_final))
        results.append(result)

    return results


def run_trotter(paras):
    qubit_wise_answers = {}
    times = np.linspace(0, paras["time"], paras["count_time"])
    h_values = np.linspace(
        10 ** paras["start_h"], 10 ** paras["end_h"], paras["count_h"]
    )

    method = paras["method"]

    if method == "lie":
        circuit_synthesis = LieCircuit
    if method == "qdrift":
        # TODO
        circuit_synthesis = LieCircuit
    else:
        circuit_synthesis = LieCircuit

    for num_qubit in range(paras["start_qubit"], paras["end_qubit"] + 1):
        observable = overall_magnetization(num_qubit)
        h_para = Parameter("h")
        parametrized_ham = parametrized_ising(num_qubit, h_para)
        circuit_manager = circuit_synthesis(parametrized_ham, h_para, paras["error"])

        h_wise_answers = {}
        for h in h_values:
            circuit_manager.subsitute_h(h)
            circuit_manager.construct_parametrized_circuit()
            ground_state = circuit_manager.ham_subbed.ground_state
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

    results = run_trotter(parameters)

    file_name = f"data/output/magnetization_{parameters['method']}_{parameters['start_qubit']}_to_{parameters['end_qubit']}.json"

    print(f"Saving results at: {file_name}")
    with open(file_name, "w") as file:
        json.dump(results, file)


def test_main():
    parameters = read_input_file("data/input/test-ising-trotter.json")

    results = run_trotter(parameters)

    file_name = f"data/output/magnetization_{parameters['method']}_{parameters['start_qubit']}_to_{parameters['end_qubit']}.json"

    print(f"Saving results at: {file_name}")
    with open(file_name, "w") as file:
        json.dump(results, file)


if __name__ == "__main__":
    main()
