import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ising.utils import read_json, read_input_file

from typing import TypeAlias

Result: TypeAlias = dict[int, dict[float, list[float]]]


def _get_point(res: Result) -> float:
    for _, h_wise_results in res.items():
        for _, result in h_wise_results.items():
            return abs(result[0])

    raise ValueError("Empty Result provided")


def _get_length(all_results: dict[str, dict[str, Result]]) -> float:
    for method, ress in all_results.items():
        for qubit, res in ress.items():
            for h, h_wise_results in res.items():
                return len(all_results) * len(ress)

    raise ValueError("Empty Result provided")


def plot_exact(paras, results: Result, color: int, **kwargs):
    times = np.linspace(0, paras["time"], paras["count_time"])
    h_value = _get_point(results)
    style = kwargs.get("style")

    for num_qubit, h_wise_results in results.items():
        sns.lineplot(x=[0], y=[h_value], alpha=0.0, label=f"N={num_qubit}")
        for h, result in h_wise_results.items():
            h_label = str(h)[:4]
            if style != "L":
                if len(times) != len(result):
                    print(h, result)
                    print(len(times), len(result))
                sns.scatterplot(
                    x=times,
                    y=result,
                    label=f"{h_label}",
                    marker=style,
                    linewidth=3,
                )
            else:
                sns.lineplot(x=times, y=result, label=f"{h_label}")


def plot_combined(
    paras, method_wise_results: dict[str, Result], diagram_name, **kwargs
):
    scale = kwargs.get("scale", (1, 1))
    styles = kwargs.get("labels", ["L"] * len(method_wise_results))
    palette = kwargs.get("palette", [])

    max_time = paras.get("time")

    for ind, (method, results) in enumerate(method_wise_results.items()):
        h_value = _get_point(results)
        sns.scatterplot(
            x=[max_time * scale[0]], y=[h_value * scale[1]], alpha=0.0, label=method
        )
        plot_exact(paras, results, style=styles[ind], color=palette[ind])

    plt.savefig(diagram_name)
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

    num_plots = _get_length(method_wise_results)
    palette = sns.blend_palette(
        ("#5a7be0", "#61c458", "#ba00b4", "#eba607", "#b30006"), n_colors=num_plots
    )

    diagram_name = (
        f"{plotfig['fig_folder']}/{method_combined}_{start_qubit}_to_{end_qubit}.png"
    )
    plot_combined(
        input_paras, method_wise_results, diagram_name, palette=palette, **plotfig
    )


if __name__ == "__main__":
    main()
