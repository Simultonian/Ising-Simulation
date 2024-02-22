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
    file_name = f"data/tgatecount/{molecule}.json"
    with open(file_name, "r") as file:
        depth = json.load(file)

    return depth

def convert_to_tick(num: str) -> str:
    power = np.log10(float(num))
    return f"$10^{{{power:.{2}f}}}$"

def plot_dictionaries(name: str, depths: dict[str, dict[float, int]]):
    fig, ax = plt.subplots()

    for method, data in depths.items():
        errors = list(data.keys())
        gates = list(data.values())
        plt.plot(errors, gates, label=method, alpha=0.6)
        plt.scatter(errors, gates)

    ax.set_xticks(errors)

    labels = [convert_to_tick(t) for t in errors]

    ax.set_xticklabels(labels)

    # Add labels for each group
    plt.xlabel("Error")
    plt.ylabel("Gate Count")
    plt.title("Method-Wise T-Gate Count vs Error for GSP")
    ax.set_yscale("log")
    plt.legend()

    file_name = f"plots/benchmark/molecule/tgatecount/{name}.png"
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
