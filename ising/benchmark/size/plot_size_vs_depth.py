import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import json

QUBITS = 15
TIME = 1.0
ORDER = 2

GATE = "cx"
FILE_NAME = "ising_one"


def _truncate(num, digits=5):
    a, dot, b = str(num).partition(".")
    return a + dot + b[:digits]


MAP = {
    "taylor": "Truncated Taylor Series",
    "qdrift": "QDrift Protocol",
    "trotter1": "First Order Trotterization",
    "trotter2": "Second Order Trotterization",
}

COLORS = ["#DC5B5A", "#625FE1", "#94E574", "#2A2A2A"]
METHODS = [
    "Truncated Taylor Series",
    "QDrift Protocol",
    "First Order Trotterization",
    "Second Order Trotterization",
]

NUMBER = {
    "Truncated Taylor Series": 0,
    "QDrift Protocol": 1,
    "First Order Trotterization": 2,
    "Second Order Trotterization": 3,
}


def _read_json(file_name):
    # The result is of the form answers[method][h][error] = num
    with open(file_name, "r") as file:
        data = json.load(file)

    new_data = {}
    for key, value in data.items():
        if key == "META":
            new_data["META"] = value
            continue

        result = {}
        for qubit, gate_count in data[key].items():
            result[float(qubit)] = float(gate_count)
        new_data[MAP[key]] = result

    return new_data


def get(data):
    method_results = data["First Order Trotterization"]
    hs = list(method_results.keys())
    errors = list(method_results[hs[0]].keys())
    return hs, errors


from ising.hamiltonian import parametrized_ising


def main():
    json_name = f"data/benchmark/size/{FILE_NAME}_{GATE}_h_eps.json"
    answers = _read_json(json_name)
    fig, ax = plt.subplots()

    # "META": {"SIZE_RANGE": f"{SIZE_RANGE[0]},{SIZE_RANGE[1]}", "TIME": str(TIME), "GATE": str(GATE)},
    # Converting the logirithmic ranges to decimals
    sizes = list(answers["Truncated Taylor Series"].keys())

    for method, label in MAP.items():
        gate_counts = list(answers[label].values())
        plt.plot(sizes, gate_counts, color=COLORS[NUMBER[label]], label=label)
        plt.scatter(sizes, gate_counts, color=COLORS[NUMBER[label]])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add labels for each group
    plt.xlabel(r"Chain Size")
    plt.ylabel(r"CNOT Gate Count ($\log_{10}$ scale)")

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=10)
    ax.set_yscale("log")

    diagram_name = f"plots/benchmark/size/{FILE_NAME}_{GATE}.png"
    print(f"Saving diagram at:{diagram_name}")
    plt.savefig(diagram_name, dpi=300)


if __name__ == "__main__":
    main()
