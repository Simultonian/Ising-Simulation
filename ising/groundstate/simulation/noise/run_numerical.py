import argparse
import json
import numpy as np

from ising.hamiltonian import parametrized_ising
from ising.utils import read_input_file

from qiskit.circuit import Parameter

from ising.hamiltonian import parametrized_ising
from ising.observables import overall_magnetization
from ising.utils import read_input_file
from ising.simulation.trotter import (
    QDriftCircuit,
    GroupedLieCircuit,
    GSQDriftCircuit,
)
from ising.simulation.exact import ExactSimulation

from ising.groundstate.simulation.noise.lcusynth import LCUNoisySynthesizer
from ising.groundstate.simulation.noise.transformer import transform_noisy_results
from ising.noise import depolarization


SYNTH_MAP = {
    "exact": ExactSimulation,
    "qdrift": QDriftCircuit,
    "grouped_lie": GroupedLieCircuit,
    "gs_qdrift": GSQDriftCircuit,
}


def run_numerical(paras):
    h_values = np.linspace(
        10 ** paras["start_h"], 10 ** paras["end_h"], paras["count_h"]
    )
    qubits = map(
        int, np.linspace(paras["start_qubit"], paras["end_qubit"], paras["qubit_count"])
    )

    circuit_synthesis = SYNTH_MAP.get(paras["method"])
    if circuit_synthesis is None:
        raise ValueError(f"Incorrect synthesis method: {paras['method']}")

    noise = paras.get("noise", "identity")
    # Result: dict[dict[qubit, dict[h, [answer]]]]
    if noise == "depolarization":
        parameters = paras.get("polarization", [0.0])
    else:
        raise ValueError("Unknown noise parameter")

    qubit_wise_answers = {}
    for num_qubit in qubits:
        observable = overall_magnetization(num_qubit)
        h_para = Parameter("h")
        parametrized_ham = parametrized_ising(num_qubit, h_para)
        circuit_manager = circuit_synthesis(parametrized_ham, h_para, paras["error"])

        if noise == "depolarization":
            noise_fns = [
                depolarization(polarization, num_qubit + 1)  # Ancilla qubit
                for polarization in parameters
            ]
        else:
            raise ValueError("Unknown noise parameter")

        h_wise_answers = {}
        for h in h_values:
            print(f"Running for {num_qubit} qubits and h:{h}")
            circuit_manager.subsitute_h(h)

            lcu_run = LCUNoisySynthesizer(
                circuit_manager,
                observable,
                overlap=paras["overlap"],
                error=paras["error"],
                success=paras["success"],
                noise=noise_fns,
            )
            h_wise_answers[h] = lcu_run.calculate_mu()

        qubit_wise_answers[num_qubit] = h_wise_answers

    return transform_noisy_results(qubit_wise_answers, parameters)


def file_numerical(file_name):
    parameters = read_input_file(file_name)

    results = run_numerical(parameters)

    file_name = f"data/groundstate/noisy_{parameters['method']}_{parameters['start_qubit']}_to_{parameters['end_qubit']}.json"

    print(f"Saving results at: {file_name}")
    with open(file_name, "w") as file:
        json.dump(results, file)


def main():
    parser = argparse.ArgumentParser(
        description="Overall magnetization of Ising using Synthesis."
    )
    parser.add_argument("--input", type=str, help="File for input parameters.")
    args = parser.parse_args()

    file_numerical(args.input)


def test_main():
    file_numerical("data/input/noisy_groundstate.json")


if __name__ == "__main__":
    main()
