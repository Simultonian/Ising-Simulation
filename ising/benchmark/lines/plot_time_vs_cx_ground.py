import os
import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns

from ising.hamiltonian import parametrized_ising_power, parametrized_ising
import json
from ising.hamiltonian import parse


QUBIT = 25
H_VAL = 0.1
ERROR_RANGE = (-1, -4)
ERROR_COUNT = 10
OVERLAP = 0.1
PROBABILITY = 0.1


# ERROR, TIME
OBS_NORM = 1
FILE_NAME = f"ising_{QUBIT}"

COLORS = ["#DC5B5A", "#625FE1", "#94E574", "#2A2A2A", "#D575EF"]
MAP = {
    "taylor": "Single Ancilla LCU",
    "qdrift": "QDrift Protocol",
    "trotter": "First Order Trotterization",
    "ktrotter": "Second Order Trotterization",
}


def test_main():
    json_name = f"data/benchmark/line/error/{FILE_NAME}_{ERROR_RANGE[0]}.json"

    with open(json_name, "r") as file:
        results = json.load(file)

    fig, ax = plt.subplots()

    for ind, (method, error_gate) in enumerate(results.items()):
        x_values, y_values = [], []
        for error_str, res_str in error_gate.items():
            error, res = float(error_str), float(res_str)
            x_values.append(error)
            y_values.append(res)

        ax = sns.lineplot(
            x=x_values,
            y=y_values,
            color=COLORS[ind],
            alpha=0.6,
        )

    for ind, (method, error_gate) in enumerate(results.items()):
        x_values, y_values = [], []
        for error_str, res_str in error_gate.items():
            error, res = float(error_str), float(res_str)
            x_values.append(error)
            y_values.append(res)

        ax = sns.scatterplot(
            x=x_values,
            y=y_values,
            label=f"{MAP[method]}",
            s=15,
            color=COLORS[ind],
        )

    # Remove the top and right border
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add labels for each group
    plt.ylabel(r"CNOT Gate Count")
    plt.xlabel(r"Error")

    # plt.title(f"Method-Wise T-Gate Count vs Error for GSP for {name}")
    ax.set_yscale("log")
    ax.set_xscale("log")

    ax.invert_xaxis()

    file_name = f"plots/benchmark/lines/size_{FILE_NAME}_{ERROR_RANGE[1]}.png"

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=10)
    plt.savefig(file_name, dpi=300)
    print(f"saved the plot to {file_name}")





if __name__ == "__main__":
    test_main()
