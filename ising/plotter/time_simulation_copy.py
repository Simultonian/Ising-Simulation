import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from ising.utils import read_json, read_input_file

from typing import TypeAlias

# Qubit -> h -> values
Result: TypeAlias = dict[str, dict[float, list[float]]]


def _get_point(res: Result) -> float:
    for _, h_wise_results in res.items():
        for _, result in h_wise_results.items():
            return abs(result[0])

    raise ValueError("Empty Result provided")


COLORS = {
    "taylor_single": "red",
    "gs_qdrift": "blue",
    "grouped_lie": "green",
    "exact": "black",
}
LABELS = {
    "taylor_single": "Truncated Taylor",
    "gs_qdrift": "qDRIFT Protocol",
    "grouped_lie": "First Order Trotter",
    "exact": "Exact Simulation",
}
MARKERS = {"taylor_single": "s", "gs_qdrift": "o", "grouped_lie": "v"}


def plot_method(method_name, paras, results: Result, **kwargs):
    times = np.linspace(0, paras["time"], paras["count_time"])
    h_value = _get_point(results)
    style = kwargs.get("style")

    for num_qubit, h_wise_results in results.items():
        for h, result in h_wise_results.items():
            if method_name in MARKERS:
                sns.scatterplot(
                    x=times,
                    y=result,
                    label=LABELS[method_name],
                    marker=MARKERS[method_name],
                    color=COLORS[method_name],
                    s=80,
                    alpha=0.8,
                )
                sns.lineplot(
                    x=times,
                    y=result,
                    color=COLORS[method_name],
                    linestyle=style,
                    alpha=0.2,
                )
            else:
                sns.lineplot(
                    x=times,
                    y=result,
                    label=LABELS[method_name],
                    color=COLORS[method_name],
                    linestyle=style,
                    linewidth=1.5,
                )


def plot_combined(
    paras, method_wise_results: dict[str, Result], diagram_name, **kwargs
):
    scale = kwargs.get("scale", (1, 1))
    styles = kwargs.get("labels", ["L"] * len(method_wise_results.items()))

    max_time = paras.get("time")

    fig, ax = plt.subplots()

    # SETTING: FONT STYLE
    plt.rcParams.update({"font.size": 12})
    plt.rcParams.update({"font.family": "sans-serif"})

    # SETTING: AXIS LENGTH
    ax.set_xlim(-1, max_time * scale[0])

    for ind, (method, results) in enumerate(method_wise_results.items()):
        # h_value = _get_point(results)
        # sns.scatterplot(
        #     x=[max_time * scale[0]], y=[h_value * scale[1]], alpha=0.0, label=method
        # )
        plot_method(method, paras, results, style=styles[ind])

    # SETTING: AXIS VISIBILITY
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # SETTING: AXIS WIDTH
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_linewidth(1.5)

    # SETTING: AXIS TICKS
    plt.locator_params(axis="x", nbins=10)
    plt.locator_params(axis="y", nbins=6)

    # SETTING: AXIS LABELS
    ax.set_xlabel("Time", fontsize=14)
    ax.set_ylabel("Magnetization M(h)", fontsize=14)

    # SETTING: TITLE PAD
    ax.set_title("Ising Model Simulation with Different Techniques", pad=20)
    # ax.get_legend().remove()

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
