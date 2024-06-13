import numpy as np
import json
import seaborn as sns
from matplotlib import pyplot as plt
from ising.hamiltonian import parametrized_ising
from ising.utils import close_state
from ising.observables import overall_magnetization
from ising.lindbladian.unraveled import lowering_all_sites, lindbladian_operator

# log scale
GAMMA_RANGE = (0, -5)
GAMMA_COUNT = 4

# not log scale
TIME_RANGE = (1, 5)
TIME_COUNT = 10

# system params
CHAIN_SIZE = 5
H_VAL = -0.1

# simulation params
OVERLAP = 0.8

COLORS = ["#DC5B5A", "#625FE1", "#94E574", "#2A2A2A", "#D575EF"]


def _truncate(num, digits=5):
    a, dot, b = str(num).partition(".")
    return a + dot + b[:digits]


def test_main():
    file_name = f"data/lindbladian/time_vs_magn_gamma/size_{CHAIN_SIZE}.json"

    with open(file_name, "r") as file:
        data = json.load(file)
    fig, ax = plt.subplots()

    for ind, (gamma_str, magn_gamma) in enumerate(data["ANSWERS"].items()):
        gamma = float(gamma_str)

        x_values, y_values = [], []
        for time_str, res_str in magn_gamma.items():
            time, res = float(time_str), float(res_str)
            x_values.append(time)
            y_values.append(res)

        ax = sns.scatterplot(
            x=x_values,
            y=y_values,
            label=rf"$\gamma$: {_truncate(gamma, digits=5)}",
            linewidth=3,
            color=COLORS[ind],
        )
        ax = sns.lineplot(
            x=x_values,
            y=y_values,
            color=COLORS[ind],
        )

    # Remove the top and right border
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add labels for each group
    plt.ylabel(r"Overall magnetization")
    plt.xlabel(r"Evolution time")

    # plt.title(f"Method-Wise T-Gate Count vs Error for GSP for {name}")
    # ax.set_yscale("log")
    # ax.set_xscale("log")

    file_name = f"plots/lindbladian/magnetization/size_{CHAIN_SIZE}.png"

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=3, fontsize=10)
    plt.savefig(file_name, dpi=300)
    print(f"saved the plot to {file_name}")


if __name__ == "__main__":
    test_main()
