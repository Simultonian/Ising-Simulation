import numpy as np
import json
import seaborn as sns
from matplotlib import pyplot as plt

QUBIT_COUNT = 1
GAMMA = 0.3
TIMES = [0, 10, 100, 150]
EPS = 0.5

H_VAL = -0.1

COLORS = ["#DC5B5A", "#625FE1", "#94E574", "#2A2A2A", "#D575EF"]
MAP = {
        "interaction": "Repeated Interactions",
        "kraus": "Kraus Operators",
        "lindbladian": "Lindbladian Operators"
    }

def _truncate(num, digits=5):
    a, dot, b = str(num).partition(".")
    return a + dot + b[:digits]

def test_main():

    file_name = f"data/lindbladian/overlap/one_qubit.json"
    with open(file_name, "r") as file:
        results = json.load(file)

    fig, ax = plt.subplots()

    x_values, y_values = [], []
    for time_str, res_str in results["lindbladian"].items():
        time, res = float(time_str), float(res_str)
        x_values.append(time)
        y_values.append(res)

    ax = sns.lineplot(
        x=x_values,
        y=y_values,
        label=f"{MAP['lindbladian']}",
        color=COLORS[0],
    )

    x_values, y_values = [], []
    for time_str, res_str in results["interaction"].items():
        time, res = float(time_str), float(res_str)
        x_values.append(time)
        y_values.append(res)

    ax = sns.scatterplot(
        x=x_values,
        y=y_values,
        label=f"{MAP['interaction']}",
        s=25,
        color=COLORS[1],
    )

    # Remove the top and right border
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add labels for each group
    plt.ylabel(r"Overlap with $|1\rangle$ state")
    plt.xlabel(r"Evolution time")

    file_name = f"plots/lindbladian/simulation/single_cm_scatter.png"

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=3, fontsize=10)
    plt.savefig(file_name, dpi=300)
    print(f"saved the plot to {file_name}")



if __name__ == "__main__":
    test_main()
