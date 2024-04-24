import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ising.hamiltonian import parametrized_ising
from ising.benchmark.sim_function import (
    taylor_gate_count,
    trotter_gate_count,
    qdrift_gate_count,
)


def plot_gate_error(qubit, h_val, err_pair, point_count_pair, obs_norm, time_pair):
    fig, ax = plt.subplots()

    configs = {
        "taylor": {"color": "blue", "label": "Truncated Taylor"},
        "trotter": {"color": "black", "label": "First Order Trotter"},
        "qdrift": {"color": "red", "label": "qDRIFT Protocol"},
    }

    # One col is fixed error
    error_points = [
        10**x for x in np.linspace(err_pair[0], err_pair[1], point_count_pair[0])
    ]
    # One row is fixed time
    time_points = [
        x for x in np.linspace(time_pair[0], time_pair[1], point_count_pair[1])
    ]

    # 2D Arrays where the first dim is time and second is error
    taylor, trotter, qdrift = [], [], []
    ham = parametrized_ising(qubit, h_val)

    nrows, ncols = len(time_points), len(error_points)
    for time in time_points:
        taylor.append([])
        trotter.append([])
        qdrift.append([])
        for error in error_points:
            taylor[-1].append(np.log2(taylor_gate_count(ham, time, error, obs_norm)))
            trotter[-1].append(np.log2(trotter_gate_count(ham, time, error)))
            qdrift[-1].append(np.log2(qdrift_gate_count(ham, time, error)))

    taylor = np.array(taylor)
    trotter = np.array(trotter)
    qdrift = np.array(qdrift)

    max_lim = np.amax(taylor)
    max_lim = max(max_lim, np.amax(trotter))
    max_lim = max(max_lim, np.amax(qdrift))

    min_lim = np.amin(taylor)
    min_lim = min(min_lim, np.amin(trotter))
    min_lim = min(min_lim, np.amin(qdrift))

    mask = np.array(nrows * [ncols * [False, True, True]], dtype=bool)
    red = np.ma.masked_where(mask, np.repeat(qdrift, 3, axis=1))

    mask = np.array(nrows * [ncols * [True, False, True]], dtype=bool)
    blue = np.ma.masked_where(mask, np.repeat(trotter, 3, axis=1))

    mask = np.array(nrows * [ncols * [True, True, False]], dtype=bool)
    green = np.ma.masked_where(mask, np.repeat(taylor, 3, axis=1))

    # SETTING THE LIMITS AND PLOTTING
    redmesh = ax.pcolormesh(red, cmap="Blues", vmax=max_lim, vmin=min_lim)
    bluemesh = ax.pcolormesh(blue, cmap="Blues", vmax=max_lim, vmin=min_lim)
    greenmesh = ax.pcolormesh(green, cmap="Blues", vmax=max_lim, vmin=min_lim)

    fig.subplots_adjust(left=0.08, bottom=0.05, right=0.86, top=0.88)

    # ADDING THE SIDE COLORBAR
    cbar = fig.colorbar(redmesh, cax=fig.add_axes([0.88, 0.05, 0.04, 0.83]))

    # TEXT ON THE COLORBAR
    cbar.ax.text(
        0.55,
        0.22,
        r"Gate Depth ($\log_{10}$ scale)",
        rotation=90,
        ha="center",
        va="center",
        transform=cbar.ax.transAxes,
        color="black",
    )

    ax.set_ylabel(r"Time ($t$)")
    ax.yaxis.set_label_coords(-0.05, 0.51)
    ax.set_xlabel(r"Error ($\log_{10}$ scale)")
    ax.xaxis.tick_top()
    ax.yaxis.tick_left()

    # SETTING THE TICK MARKS
    x_ticks = [x * 3 for x in list(range(1, point_count[0] + 1))]
    ax.set_xticks(x_ticks)
    ax.set_yticks(list(range(point_count[1])))
    ax.tick_params(direction="out")

    # SETTING THE LABELS
    error_marks = [np.log10(x) for x in error_points]
    ax.set_xticklabels(["{}".format(i) for i in error_marks])
    ax.set_yticklabels(["{}".format(i) for i in time_points])

    # LABELLING the BARS
    ax.text(0.5, 0.35, "qDRIFT", rotation=90, ha="center", va="center", color="black")
    ax.text(
        1.5,
        1.09,
        "First Order Trotterization",
        rotation=90,
        ha="center",
        va="center",
        color="black",
    )
    ax.text(
        2.5,
        1.05,
        "Truncated Taylor Series",
        rotation=90,
        ha="center",
        va="center",
        color="black",
    )

    for x_stamp in x_ticks:
        ax.axvline(x=x_stamp, color="black", linestyle="-")

    diagram_name = "plots/benchmark/heat_plot.png"
    print(f"Saving diagram at:{diagram_name}")
    plt.savefig(diagram_name, dpi=300)


if __name__ == "__main__":
    qubit = 25
    h_val = 0.1
    err_pair = (-1, -5)
    point_count = (5, 5)
    obs_norm = 1
    time_pair = (1, 20)
    plot_gate_error(qubit, h_val, err_pair, point_count, obs_norm, time_pair)
