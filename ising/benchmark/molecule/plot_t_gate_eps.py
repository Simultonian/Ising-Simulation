import numpy as np
import json
from matplotlib import pyplot as plt
from ising.hamiltonian import Hamiltonian, parse, parametrized_ising

colors: dict[str, str] = {
    "Truncated Taylor Series": "blue",
    "First Order Trotter": "black",
    "qDRIFT": "red",
}

markers: dict[str, str] = {
    "Truncated Taylor Series": "o",
    "First Order Trotter": "^",
    "qDRIFT": "x",
}


def get_gate_count_gsp(molecule: str):
    file_name = f"data/tgatecount/{molecule}.json"
    with open(file_name, "r") as file:
        depth = json.load(file)

    return depth

def convert_to_tick(num: str) -> str:
    power = np.log10(float(num))
    return f"${{{power:.{2}f}}}$"

def plot_dictionaries(name: str, depths: dict[str, dict[float, int]]):
    fig, ax = plt.subplots()

    for method, data in depths.items():
        errors = list(data.keys())
        gates = list(data.values())
        plt.plot(errors, gates, color=colors[method], alpha=0.6)
        plt.scatter(errors, gates, color=colors[method], label=method, marker=markers[method])

    ax.set_xticks(errors)

    labels = [convert_to_tick(t) for t in errors]

    ax.set_xticklabels(labels)

    # Remove the top and right border
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add labels for each group
    plt.xlabel(r"$\log(\text{Error})$")
    plt.ylabel(r"T Gate Count ($\log$ scale)")
    # plt.title(f"Method-Wise T-Gate Count vs Error for GSP for {name}")
    ax.set_yscale("log")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=3, fontsize=10)

    file_name = f"plots/benchmark/molecule/tgatecount/{name}.png"
    plt.savefig(file_name, dpi=300)
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
