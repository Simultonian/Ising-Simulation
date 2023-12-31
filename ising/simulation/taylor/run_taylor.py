import argparse
import json
import numpy as np

from qiskit.circuit import Parameter

from ising.hamiltonian import parametrized_ising
from ising.observables import overall_magnetization
from ising.utils import read_input_file, close_state
from ising.simulation.taylor import TaylorSample, Taylor
from ising.utils.constants import PLUS


SYNTH = {
    "taylor_sample": TaylorSample,
    "taylor_single": Taylor,
}


def run_taylor(paras):
    qubit_wise_answers = {}
    times = np.linspace(
        paras["time"] / paras["count_time"], paras["time"], paras["count_time"]
    )
    h_values = np.linspace(
        10 ** paras["start_h"], 10 ** paras["end_h"], paras["count_h"]
    )

    method = paras["method"]
    if method not in SYNTH:
        raise ValueError("This is Taylor file, method called:", method)

    synth = SYNTH[method]

    qubits = map(
        int, np.linspace(paras["start_qubit"], paras["end_qubit"], paras["qubit_count"])
    )

    for num_qubit in qubits:
        h_para = Parameter("h")
        parametrized_ham = parametrized_ising(num_qubit, h_para)
        circuit_manager = synth(
            parametrized_ham, h_para, paras["error"], success=paras.get("success", None)
        )

        circuit_manager.substitute_obs(overall_magnetization(num_qubit))
        h_wise_answers = {}
        for h in h_values:
            print(f"Running for {num_qubit} qubits and h:{h}")
            circuit_manager.subsitute_h(h)

            ground_state = circuit_manager.ground_state

            # Taking tensor product of the overlap state with `|+>` which is the state after `H`
            init_state = np.kron(
                PLUS, close_state(ground_state, paras["overlap"])
            ).reshape(-1, 1)

            # Checking for norm
            np.testing.assert_almost_equal(np.sum(np.abs(init_state) ** 2), 1)

            circuit_manager.set_up_decomposition(max(times))
            ans = circuit_manager.get_observations(init_state, times)
            h_wise_answers[h] = ans

        qubit_wise_answers[num_qubit] = h_wise_answers

    return qubit_wise_answers


def file_taylor(file_name):
    parameters = read_input_file(file_name)
    results = run_taylor(parameters)

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
    file_taylor(args.input)


def test_main():
    file_taylor("data/input/taylor.json")


if __name__ == "__main__":
    np.random.seed(42)
    main()
