from typing import TypeAlias

import argparse
import math
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from ising.utils import read_json, read_input_file


# Qubit -> h -> magn
Result: TypeAlias = dict[int, dict[float, float]]


def get_point(res: Result) -> float:
    for _, h_wise_results in res.items():
        for _, result in h_wise_results.items():
            return abs(result)

    raise ValueError("Empty Result provided")


def get_length(all_results: dict[str, Result]) -> int:
    for method, ress in all_results.items():
        return len(all_results) * len(ress)

    raise ValueError("Empty Result provided")


def log_label_maker(values: list[float]) -> list[str]:
    return ["{:.2f}".format(math.trunc(np.log(float(x)) * 100) / 100) for x in values]


def plot_one(results: Result, color: int, **kwargs):
    style = kwargs.get("style")

    for num_qubit, h_wise_results in results.items():
        # h_wise_results: h -> magn
        y_values = list(h_wise_results.values())
        x_values = log_label_maker(list(h_wise_results.keys()))
        if style != "L":
            ax = sns.scatterplot(
                x=x_values,
                y=y_values,
                label=f"{num_qubit}",
                marker=style,
                linewidth=3,
            )
        else:
            ax = sns.lineplot(x=x_values, y=y_values, label=f"{num_qubit}")


def plot_combined(
    paras, method_wise_results: dict[str, Result], diagram_name, **kwargs
):
    scale = kwargs.get("scale", (1, 1))
    styles = kwargs.get("labels", ["L"] * len(method_wise_results))
    palette = kwargs.get("palette", [])

    max_h = 10 ** paras.get("end_h")

    for ind, (method, results) in enumerate(method_wise_results.items()):
        h_value = get_point(results)
        sns.scatterplot(
            x=[max_h * scale[0]], y=[h_value * scale[1]], alpha=0.0, label=method
        )
        plot_one(results, color=palette[ind], style=styles[ind])

    plt.savefig(diagram_name)
    print(f"Saving diagram at {diagram_name}")


def main():
    parser = argparse.ArgumentParser(description="Phase transition of Ising model")
    parser.add_argument("--input", type=str, help="File for input parameters.")
    args = parser.parse_args()
    plotfig = read_json(args.input)

    method_input_file = plotfig["input_file"]
    input_paras = read_input_file(method_input_file)

    start_qubit, end_qubit = input_paras["start_qubit"], input_paras["end_qubit"]
    observable = plotfig["observable"]

    method_wise_results = {}
    for method in plotfig["methods"]:
        method_output_file = f"data/singlelcu/output/{observable}_{method}_{start_qubit}_to_{end_qubit}.json"
        method_wise_results[method] = read_json(method_output_file)

    num_plots = get_length(method_wise_results)
    palette = sns.blend_palette(
        ("#5a7be0", "#61c458", "#ba00b4", "#eba607", "#b30006"), n_colors=num_plots
    )

    method_combined = "_".join(plotfig["methods"])
    diagram_name = (
        f"{plotfig['fig_folder']}/{method_combined}_{start_qubit}_to_{end_qubit}.png"
    )
    plot_combined(
        input_paras, method_wise_results, diagram_name, palette=palette, **plotfig
    )


if __name__ == "__main__":
    main()
