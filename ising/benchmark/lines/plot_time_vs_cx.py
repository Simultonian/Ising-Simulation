import matplotlib.pyplot as plt
import seaborn as sns

import json


QUBIT = 25
H_VAL = 0.1
ERROR = 0.01
DELTA = 0.1

# ERROR, TIME
OBS_NORM = 1
TIME_PAIR = (0, 2)
TIME_COUNT = 8
FILE_NAME = f"ising_{QUBIT}"

COLORS = ["#DC5B5A", "#625FE1", "#94E574", "#2A2A2A", "#D575EF"]
MAP = {
    "taylor": "Truncated Taylor Series",
    "qdrift": "QDrift Protocol",
    "trotter": "First Order Trotterization",
    "ktrotter": "Second Order Trotterization",
}


def test_main():
    json_name = f"data/benchmark/line/time/{FILE_NAME}.json"

    with open(json_name, "r") as file:
        results = json.load(file)

    fig, ax = plt.subplots()

    for ind, (method, time_gate) in enumerate(results.items()):
        x_values, y_values = [], []
        for time_str, res_str in time_gate.items():
            time, res = float(time_str), float(res_str)
            x_values.append(time)
            y_values.append(res)

        ax = sns.lineplot(
            x=x_values,
            y=y_values,
            color=COLORS[ind],
            alpha=0.6,
        )

    for ind, (method, time_gate) in enumerate(results.items()):
        x_values, y_values = [], []
        for time_str, res_str in time_gate.items():
            time, res = float(time_str), float(res_str)
            x_values.append(time)
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
    plt.xlabel(r"Evolution time")

    # plt.title(f"Method-Wise T-Gate Count vs Error for GSP for {name}")
    ax.set_yscale("log")
    ax.set_xscale("log")

    file_name = f"plots/benchmark/lines/size_{FILE_NAME}.png"

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=10)
    plt.savefig(file_name, dpi=300)
    print(f"saved the plot to {file_name}")


if __name__ == "__main__":
    test_main()
