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
    "taylor_single": "Truncated Taylor Series",
}


def _get_point(res: dict[float, float]) -> tuple[float, float]:
    for error, result in res.items():
        return error, result

    raise ValueError("Empty Result provided")


def log_label_maker(values: list[float]) -> list[str]:
    return ["{:.2f}".format(math.trunc(np.log(float(x)) * 100) / 100) for x in values]


def calculate_method_error(
    target: dict[float, dict[float, float]], result: NoisyResult
) -> dict[float, float]:
    """
    Calculates just the difference for all points and takes the average.
    # Noise -> Qubit -> h -> values
    """
    final = {}
    for noise, qubit_wise in result.items():
        count, sm = 0, 0.0
        for qubit, h_wise in qubit_wise.items():
            for hval, res in h_wise.items():
                targ = target[qubit][hval]
                sm += abs(res - targ)
                count += 1

        final[noise] = sm / count

    return final


def process_error(
    method_wise_results: dict[str, NoisyResult]
) -> dict[str, dict[float, float]]:
    """
    Converts the method wise results into the error for each parameter of
    the noise for each method.
    """
    error_bounds = {}
    target = method_wise_results.get("analytical", None)
    if target is None:
        raise ValueError("Can't find analytical in results.")

    for method, result in method_wise_results.items():
        if method == "analytical":
            continue
        error_bounds[method] = calculate_method_error(target, result)

    return error_bounds


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
    fig, ax = plt.subplots()

    error_points = process_error(method_wise_results)

    # SETTING: FONT STYLE
    plt.rcParams.update({"font.size": 12})
    plt.rcParams.update({"font.family": "sans-serif"})

    for ind, (method, noisy_results) in enumerate(error_points.items()):
        if method != "analytical":
            err, res = _get_point(noisy_results)

            noise_x = [float(x) for x in list(noisy_results.keys())]
            error_y = list(noisy_results.values())

            ax = sns.scatterplot(
                x=noise_x,
                y=error_y,
                marker="x",
                linewidth=3,
                # color=color,
            )
            ax = sns.lineplot(
                x=noise_x,
                y=error_y,
                label=METHOD_NAMES[method],
                linewidth=2,
                alpha=0.8
                # color=color,
            )

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
    ax.set_xlabel(r"Depolarization Noise $p$ ($\log$ scale)", fontsize=14)
    ax.set_ylabel("$|M_{E} - M_{S}|$", fontsize=14)

    # SETTING: TITLE PAD
    # ax.set_title("Error due to Noise", pad=20)
    ax.get_legend()

    # SETTING: LOG SCALE
    ax.set_xscale("log")
    # ax.set_yscale("log")

    # SETTING: INVERSION
    ax.invert_xaxis()

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

    diagram_name = f"plots/groundstate/noise_gap_{method_combined}_{start_qubit}_to_{end_qubit}.png"
    plot_combined(input_paras, method_wise_results, diagram_name, **plotfig)


def main():
    parser = argparse.ArgumentParser(description="Overall magnetization of Ising")
    parser.add_argument("--input", type=str, help="File for input parameters.")
    args = parser.parse_args()
    file_run(args.input)


def test_main():
    file_run("data/plotfig/noisy_groundstate_taylor.json")


if __name__ == "__main__":
    main()
