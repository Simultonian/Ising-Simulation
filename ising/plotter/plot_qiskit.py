import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ising.utils import read_input_file
import json


def read_exact_output(file_name):
    with open(file_name, "r") as file:
        json_str = file.read()
        data = json.loads(json_str)

    return data


def plot_exact(paras, results: dict[int, dict[float, list[float]]], diagram_name):
    times = np.linspace(0, paras["time"], paras["count_time"])
    for num_qubit, h_wise_results in results.items():
        h_value = list(h_wise_results.items())[0][1][0]
        sns.scatterplot(x=[0], y=[h_value], alpha=0.0, label=f"N={num_qubit}")
        for h, result in h_wise_results.items():
            h_label = str(h)[:4]
            sns.lineplot(x=times, y=result, label=f"{h_label}")

    plt.savefig(diagram_name)
    print(f"Saving diagram at {diagram_name}")


def main():
    parser = argparse.ArgumentParser(description="Overall magnetization of Ising")
    parser.add_argument("--input", type=str, help="File for input parameters.")
    args = parser.parse_args()
    parameters = read_input_file(args.input)

    output_file = f"data/output/ising_qiskit_trotter_{parameters['start_qubit']}_to_{parameters['end_qubit']}.json"
    results = read_exact_output(output_file)

    diagram_name = f"plots/qiskit_trotter/magnetization_{parameters['start_qubit']}_to_{parameters['end_qubit']}.png"
    plot_exact(parameters, results, diagram_name)


if __name__ == "__main__":
    main()
