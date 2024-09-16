import matplotlib.pyplot as plt
import seaborn as sns

import json


QUBIT_COUNT = 5

COLORS = ["#DC5B5A", "#625FE1", "#94E574", "#2A2A2A", "#D575EF"]
MAP = {
    "sal": "Single Ancilla LCU",
    "qdrift": "QDrift Protocol",
    "trotter": "First Order Trotterization",
    "ktrotter": "Second Order Trotterization",
}


def test_main():
    file_name = f"data/benchmark/line/lindi/error_size_{QUBIT_COUNT}.json"

    with open(file_name, "r") as file:
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

    file_name = f"plots/benchmark/lines/lindi/no_label_error_size_{QUBIT_COUNT}.png"

    ax.get_legend().remove()
    # plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=10)
    plt.savefig(file_name, dpi=300)
    print(f"saved the plot to {file_name}")


if __name__ == "__main__":
    test_main()
