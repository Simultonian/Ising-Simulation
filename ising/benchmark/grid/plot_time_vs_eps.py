import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import json


# log scale
TIME_RANGE = (0, 2)
TIME_COUNT = 20

QUBITS = 15
TIME = 1.0
ORDER = 2

ERROR_RANGE = (-1, -5)
ERROR_COUNT = 5
GATE = "cx"
FILE_NAME = "methane"


def _truncate(num, digits=5):
    a, dot, b = str(num).partition(".")
    return a + dot + b[:digits]


def _truncate_error(num, digits=5):
    x = int(np.log10(num))
    return f"1e{x}"


MAP = {
    "taylor": "Truncated Taylor Series",
    "qdrift": "QDrift Protocol",
    "trotter": "First Order Trotterization",
    "ktrotter": "Second Order Trotterization",
}

COLORS = ["#DC5B5A", "#EAEAEB", "#94E574", "#2A2A2A"]
LABEL_COLORS = ["#DC5B5A", "#2A2A2A"]
LABEL_METHODS = [
    "Truncated Taylor Series",
    "Second Order Trotterization",
]

METHODS = [
    "Truncated Taylor Series",
    "QDrift Protocol",
    "First Order Trotterization",
    "Second Order Trotterization",
]

NUMBER = {
    "Truncated Taylor Series": 0,
    "QDrift Protocol": 1,
    "First Order Trotterization": 2,
    "Second Order Trotterization": 3,
}


def _read_json(file_name):
    # The result is of the form answers[method][h][error] = num
    with open(file_name, "r") as file:
        data = json.load(file)

    new_data = {}
    for key, value in data.items():
        if key == "META":
            new_data["META"] = value
            continue

        result = []
        for row in value:
            converted_row = [float(num) for num in row]
            result.append(converted_row)
        new_data[MAP[key]] = result

    return new_data


from ising.hamiltonian import parametrized_ising


def main():
    json_name = f"data/benchmark/heat/{FILE_NAME}_gate_err_data.json"
    answers = _read_json(json_name)

    # One col is fixed error
    error_points = [
        10**x for x in np.linspace(ERROR_RANGE[0], ERROR_RANGE[1], ERROR_COUNT)
    ]
    min_error = min(error_points)

    # One row is fixed time
    time_points = [
        10**x for x in np.linspace(TIME_RANGE[0], TIME_RANGE[1], TIME_COUNT)
    ]

    best_method = {}
    best_gate_count = {}
    mx_gate, mn_gate = 0, float("inf")
    color_grid = []
    for time_ind, time in enumerate(time_points):
        best_method[time] = {}
        best_gate_count[time] = {}
        row = []

        for error_ind, error in enumerate(error_points):
            cur_min, cur_method = float("inf"), None
            for method in MAP.values():
                cur = answers[method][time_ind][error_ind]
                mx_gate = max(cur, mx_gate)
                mn_gate = min(cur, mn_gate)
                if cur < cur_min:
                    cur_min = cur
                    cur_method = method

            assert cur_method is not None
            best_method[time][error] = NUMBER[cur_method]
            best_gate_count[time][error] = cur_min
            row.append(NUMBER[cur_method])

        color_grid.append(row)

    cmap = sns.color_palette(COLORS)

    fig, ax = plt.subplots()
    sns.heatmap(color_grid, cmap=cmap, cbar=False, linewidths=0.5, linecolor="gray")

    plt.xlabel("Error")
    plt.ylabel("Time")

    error_labels = [_truncate_error(x, 4) for x in error_points]
    time_labels = [_truncate(x, 3) for x in time_points]

    print(error_labels)
    print(time_labels)
    plt.xticks(np.arange(0.5, 5.5, 1), error_labels, rotation=0)
    plt.yticks(np.arange(0.5, 20.5), time_labels, rotation=0)

    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=LABEL_COLORS[i],
            markersize=10,
            label=method,
        )
        for i, method in enumerate(LABEL_METHODS)
    ]
    fig.subplots_adjust(left=0.13, bottom=0.10, right=0.85, top=0.85)
    plt.legend(handles=legend_elements, loc=(-0.0, 1.04), ncol=2)

    diagram_name = f"plots/benchmark/grid/{FILE_NAME}_{GATE}_time_vs_eps.png"
    print(f"Saving diagram at:{diagram_name}")
    plt.savefig(diagram_name, dpi=300)


if __name__ == "__main__":
    main()
