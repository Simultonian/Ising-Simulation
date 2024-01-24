import argparse
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from ising.utils import read_json, read_input_file
from typing import TypeAlias

# Noise -> Qubit -> h -> values
NoisyResult: TypeAlias = dict[float, dict[float, dict[float, float]]]
# Qubit -> h -> values
Result: TypeAlias = dict[float, dict[float, float]]


METHOD_NAMES = {
    "analytical": "Exact Solution",
    "taylor": "Truncated Taylor Series",
    "grouped_lie": "First Order Trotterization",
    "gs_qdrift": "qDRIFT Protocol",
    "exact": "Precise Unitaries",
    "taylor_sample": "Truncated Taylor Series Samples",
    "taylor_single": "Truncated Taylor Series Optimized",
}


def _get_point(res: NoisyResult) -> float:
    for _, qubit_wise_results in res.items():
        for _, h_wise_results in qubit_wise_results.items():
            for _, result in h_wise_results.items():
                return abs(result)

    raise ValueError("Empty Result provided")


def log_label_maker(values: list[float]) -> list[str]:
    return ["{:.2f}".format(math.trunc(np.log(float(x)) * 100) / 100) for x in values]


def plot_method(paras, results: Result, **kwargs):
    style = kwargs.get("style")
    color = kwargs.get("color")
    label = kwargs.get("label")

    for num_qubit, h_wise_results in results.items():
        # h_wise_results: h -> magn
        y_values = list(h_wise_results.values())
        x_values = log_label_maker(list(h_wise_results.keys()))
        if style != "L":
            ax = sns.scatterplot(
                x=x_values,
                y=y_values,
                label=label,
                marker=style,
                linewidth=3,
                # color=color,
            )
        else:
            ax = sns.lineplot(
                x=x_values, y=y_values, label=f"N={num_qubit}", color=color
            )


def plot_combined(
    paras, method_wise_results: dict[str, NoisyResult], diagram_name, **kwargs
):
    scale = kwargs.get("scale", (1, 1))
    styles = kwargs.get("labels", ["L"] * len(method_wise_results))
    colors = kwargs.get("colors", [None] * len(method_wise_results))

    max_h = 10 ** paras.get("end_h")

    fig, ax = plt.subplots()

    # SETTING: FONT STYLE
    plt.rcParams.update({"font.size": 12})
    plt.rcParams.update({"font.family": "sans-serif"})

    for ind, (method, noisy_results) in enumerate(method_wise_results.items()):
        if method != "analytical":
            h_value = _get_point(noisy_results)
            sns.scatterplot(
                x=[max_h * scale[0]],
                y=[h_value * scale[1]],
                alpha=0.0,
                label=METHOD_NAMES[method],
            )
            for noise, results in noisy_results.items():
                if noise != "0.0":
                    label = f"Noise = {noise}"
                else:
                    label = "Noiseless"

                plot_method(
                    paras,
                    results,
                    style=styles[ind],
                    color=colors[ind],
                    label=label,
                )

        else:
            h_value = _get_point({0.0: noisy_results})
            sns.scatterplot(
                x=[max_h * scale[0]],
                y=[h_value * scale[1]],
                alpha=0.0,
                label=METHOD_NAMES[method],
            )
            plot_method(paras, noisy_results, style=styles[ind], color=colors[ind])

    # SETTING: AXIS VISIBILITY
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # SETTING: AXIS WIDTH
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_linewidth(1.5)

    # SETTING: AXIS TICKS
    # plt.locator_params(axis="x", nbins=10)
    # plt.locator_params(axis="y", nbins=6)

    # SETTING: AXIS LABELS
    ax.set_xlabel("External Magnetic Field $h$", fontsize=14)
    ax.set_ylabel("Magnetization M(h)", fontsize=14)

    # SETTING: TITLE PAD
    ax.set_title("Phase Transition of Ising Model", pad=20)
    ax.get_legend()

    # SETTING: LOG SCALE
    # ax.set_xscale("log")

    plt.savefig(diagram_name, dpi=300)
    print(f"Saving diagram at {diagram_name}")


def file_run(input_file):
    plotfig = read_json(input_file)
    method_input_file = plotfig["input_file"]
    input_paras = read_input_file(method_input_file)
    start_qubit, end_qubit = input_paras["start_qubit"], input_paras["end_qubit"]
    method_wise_results = {}

    method_combined = "_".join(plotfig["methods"])

    for method in plotfig["methods"]:
        if method != "analytical":
            method_output_file = (
                f"data/groundstate/noisy_{method}_{start_qubit}_to_{end_qubit}.json"
            )
        else:
            method_output_file = (
                f"data/groundstate/analytical_{start_qubit}_to_{end_qubit}.json"
            )

        results = read_json(method_output_file)
        method_wise_results[method] = results

    diagram_name = (
        f"plots/groundstate/noisy_{method_combined}_{start_qubit}_to_{end_qubit}.png"
    )
    plot_combined(input_paras, method_wise_results, diagram_name, **plotfig)


def main():
    parser = argparse.ArgumentParser(description="Overall magnetization of Ising")
    parser.add_argument("--input", type=str, help="File for input parameters.")
    args = parser.parse_args()
    file_run(args.input)


def test_main():
    file_run("data/plotfig/groundstate_noisy.json")


if __name__ == "__main__":
    main()
