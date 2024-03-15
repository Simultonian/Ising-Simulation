import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from ising.utils import read_json, read_input_file

from typing import TypeAlias

# Qubit -> h -> values
Result: TypeAlias = dict[str, dict[float, list[float]]]


method_colors = {
        "exact": "black",
        "taylor_single": "blue",
        "grouped_lie": "green",
        "gs_qdrift": "red",
    }

method_label = {
        "exact": "Exact Simulation",
        "taylor_single": "Truncated Taylor Series",
        "grouped_lie": "First Order Trotterization",
        "gs_qdrift": "qDRIFT Protocol"
    }

def _get_point(res: Result) -> float:
    for _, h_wise_results in res.items():
        for _, result in h_wise_results.items():
            return abs(result[0])

    raise ValueError("Empty Result provided")


def plot_method(method, paras, results: Result, **kwargs):
    times = np.linspace(0, paras["time"], paras["count_time"])
    h_value = _get_point(results)
    style = kwargs.get("style")

    for num_qubit, h_wise_results in results.items():
        for h, result in h_wise_results.items():
            h_label = str(h)[:4]
            if style != "L":
                sns.scatterplot(
                    x=times,
                    y=result,
                    label=method_label[method],
                    marker=style,
                    s=70,
                    alpha=0.8,
                    # linewidth=3,
                    color=method_colors[method],
                )
            else:
                sns.scatterplot(
                    x=times,
                    y=result,
                    marker="o",
                    s=20,
                    color=method_colors[method],
                    label=method_label[method],
                )
                sns.lineplot(
                    x=times, y=result, color=method_colors[method],
                    alpha=0.7
                )


                error = [0.1 * x for x in result]
                # ADDING ERROR BAR
                plt.errorbar(
                        x=times, 
                        y=result, 
                        yerr=error, 
                        fmt='.', 
                        alpha=0.8,
                        color=method_colors[method],
                        capsize=6,
                        )


def plot_combined(
    paras, method_wise_results: dict[str, Result], diagram_name, **kwargs
):
    scale = kwargs.get("scale", (1, 1))
    styles = kwargs.get("labels", ["L"] * len(method_wise_results))

    max_time = paras.get("time")

    fig, ax = plt.subplots()

    # SETTING: FONT STYLE
    plt.rcParams.update({"font.size": 12})
    plt.rcParams.update({"font.family": "sans-serif"})

    # SETTING: AXIS LENGTH
    # ax.set_xlim(-1, max_time * scale[0])

    for ind, (method, results) in enumerate(method_wise_results.items()):
        h_value = _get_point(results)
        plot_method(method, paras, results, style=styles[ind])

    # SETTING: AXIS VISIBILITY
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # SETTING: AXIS WIDTH
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_linewidth(1.5)

    # SETTING: AXIS TICKS

    ax.yaxis.set_ticks(np.arange(0.55, 0.81, 0.05))
    plt.locator_params(axis="x", nbins=10)
    # plt.locator_params(axis="y", nbins=6)

    # SETTING: AXIS LABELS
    ax.set_xlabel("Time $t$", fontsize=14)
    ax.set_ylabel("Magnetization $M_z$", fontsize=14)

    # SETTING: TITLE PAD
    # ax.set_title("Truncated Taylor Series", pad=20)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=2, fontsize=10)

    plt.savefig(diagram_name, dpi=300)
    print(f"Saving diagram at {diagram_name}")


def main():
    parser = argparse.ArgumentParser(description="Overall magnetization of Ising")
    parser.add_argument("--input", type=str, help="File for input parameters.")
    args = parser.parse_args()
    plotfig = read_json(args.input)

    method_input_file = plotfig["input_file"]
    input_paras = read_input_file(method_input_file)
    start_qubit, end_qubit = input_paras["start_qubit"], input_paras["end_qubit"]
    method_wise_results = {}

    method_combined = "_".join(plotfig["methods"])
    observable = plotfig["observable"]

    for method in plotfig["methods"]:
        method_output_file = (
            f"data/simulation/{observable}_{method}_{start_qubit}_to_{end_qubit}.json"
        )
        results = read_json(method_output_file)
        method_wise_results[method] = results

    diagram_name = (
        f"{plotfig['fig_folder']}/{method_combined}_{start_qubit}_to_{end_qubit}.png"
    )
    plot_combined(input_paras, method_wise_results, diagram_name, **plotfig)


if __name__ == "__main__":
    main()
