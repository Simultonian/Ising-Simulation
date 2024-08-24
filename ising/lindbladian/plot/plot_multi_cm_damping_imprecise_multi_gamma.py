import numpy as np
import json


import seaborn as sns
from matplotlib import pyplot as plt

COLORS = ["#DC5B5A", "#625FE1", "#94E574", "#2A2A2A", "#D575EF"]

QUBIT_COUNT = 3
INV_TEMPS = [1, 10]

def _round(mat):
    return np.round(mat, decimals=2)


def _get_value_time_order(results, inv_temp, times, label):
    vals = []
    for time in times:
        vals.append(results[label][time][str(inv_temp)])
    return vals

def test_main():
    np.random.seed(42)


    file_name = f"data/lindbladian/time_vs_magn_gamma/size_{QUBIT_COUNT}.json"
    with open(file_name, "r") as file:
        results = json.load(file)

    time_labels = list(results["interaction"].keys())
    for ind, inv_temp in enumerate(INV_TEMPS):
        lindbladian = _get_value_time_order(results, inv_temp, time_labels, "lindbladian")
        sal = _get_value_time_order(results, inv_temp, time_labels, "sal")

        times = [_round(float(time_str)) for time_str in time_labels]

        ax = sns.lineplot(
            x=times,
            y=lindbladian,
            label=f"Lindbladian {inv_temp}",
            color=COLORS[ind],
        )
        ax = sns.scatterplot(
            x=times,
            y=sal,
            label=f"Single Ancilla LCU {inv_temp}",
            s=35,
            color=COLORS[ind],
        )

    # Remove the top and right border
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add labels for each group
    plt.ylabel(r"Overall Magnetization")
    plt.xlabel(r"Evolution time")

    file_name = f"plots/lindbladian/simulation/size_{QUBIT_COUNT}_multi_cm_magn_gamma.png"

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=3, fontsize=10)
    plt.savefig(file_name, dpi=300)
    print(f"saved the plot to {file_name}")


if __name__ == "__main__":
    test_main()
