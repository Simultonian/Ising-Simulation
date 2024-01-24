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
from ising.noise import depolarization


SYNTH_MAP = {
    "exact": ExactSimulation,
    "qdrift": QDriftCircuit,
    "grouped_lie": GroupedLieCircuit,
    "gs_qdrift": GSQDriftCircuit,
}


def run_numerical(paras):
    qubit_wise_answers = {}
    h_values = np.linspace(
        10 ** paras["start_h"], 10 ** paras["end_h"], paras["count_h"]
    )

    noise = paras.get("noise", "identity")
    if noise == "depolarization":
        polarization_lst = paras.get("polarization", [0.0])
        noise_fns = [lambda qubits: depolarization(polarization, qubits) for polarization in polarization_lst]
    elif noise == "identity":
        noise_fns = [lambda x: x]
    else:
        raise ValueError(f"Unknown noise channel: {noise}")

    qubits = map(
        int, np.linspace(paras["start_qubit"], paras["end_qubit"], paras["qubit_count"])
    )

    circuit_synthesis = SYNTH_MAP.get(paras["method"])
    if circuit_synthesis is None:
        raise ValueError(f"Incorrect synthesis method: {paras['method']}")

    for num_qubit in qubits:
        observable = overall_magnetization(num_qubit)
        h_para = Parameter("h")
        parametrized_ham = parametrized_ising(num_qubit, h_para)
        circuit_manager = circuit_synthesis(parametrized_ham, h_para, paras["error"])
        noise_lst = [noise_fn(num_qubit + 1) for noise_fn in noise_fns] # one ancilla qubit

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
                noise=noise_lst
            )

            h_noisy_answers = lcu_run.calculate_mu()

            h_wise_answers[h] = lcu_run.calculate_mu()

        qubit_wise_answers[num_qubit] = h_wise_answers

    return qubit_wise_answers


def file_numerical(file_name):
    parameters = read_input_file(file_name)

    results = run_numerical(parameters)

    file_name = f"data/groundstate/{parameters['method']}_{parameters['start_qubit']}_to_{parameters['end_qubit']}.json"

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
    file_numerical("data/input/groundstate.json")


if __name__ == "__main__":
    main()
