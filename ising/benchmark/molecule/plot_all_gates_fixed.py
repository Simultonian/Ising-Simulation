import numpy as np
import json
from matplotlib import pyplot as plt
from ising.hamiltonian import Hamiltonian, parse, parametrized_ising

colors: dict[str, str] = {
    "Truncated Taylor Series": "blue",
    "First Order Trotter": "purple",
    "qDRIFT": "yellow",
}


def get_gate_count_gsp(molecule: str):
    file_name = f"data/gatecount/{molecule}.json"
    with open(file_name, "r") as file:
        depth = json.load(file)

    return depth


def plot_dictionaries(name: str, dicts: dict[str, dict[str, int]]):
    gates = list(dicts["qDRIFT"].keys())

    x = np.arange(len(gates))

    # Define the width of the bars and the spaces between them
    bar_width = 0.25

    # Plot each dictionary
    rects1 = plt.bar(
        x - bar_width,
        dicts["First Order Trotter"].values(),
        bar_width,
        label="First Order Trotter",
    )
    rects2 = plt.bar(x, dicts["qDRIFT"].values(), bar_width, label="qDRIFT")
    rects3 = plt.bar(
        x + bar_width,
        dicts["Truncated Taylor Series"].values(),
        bar_width,
        label="Truncated Taylor Series",
    )

    # Add labels for each group
    plt.xlabel("Gates")
    plt.ylabel("Gate Count")
    plt.title("Method-Wise Gate Count for GSP")
    plt.xticks(x, gates)
    plt.legend()

    file_name = f"plots/benchmark/molecule/gatecount/{name}.png"
    plt.savefig(file_name)
    print(f"Saving at {file_name}")


def main():
    name = "ising"
    gate_dicts = get_gate_count_gsp(name)
    print(gate_dicts)
    plot_dictionaries(name, gate_dicts)


def test_main():
    main()


if __name__ == "__main__":
    main()
