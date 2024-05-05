import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import json

QUBITS = 15
TIME = 1.0
ORDER = 2

# THE BELOW ARE LOGARITHMIC WRT 10
H_RANGE = (-3, 2)
H_COUNT = 10

ERROR_RANGE = (-1, -3)
ERROR_COUNT = 10
GATE = "cx"
FILE_NAME = "ising_one"


def _truncate(num, digits=5):
    a, dot, b = str(num).partition(".")
    return a + dot + b[:digits]


MAP = {
    "taylor": "Truncated Taylor Series",
    "qdrift": "QDrift Protocol",
    "trotter1": "First Order Trotterization",
    "trotter2": "Second Order Trotterization",
}

COLORS = ["#DC5B5A", "#EAEAEB", "#94E574", "#2A2A2A"]
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

        result = {}
        for h, ans in data[key].items():
            result[float(h)] = {}
            for err, gate_count in ans.items():
                result[float(h)][float(err)] = float(gate_count)
        new_data[MAP[key]] = result

    return new_data


def get_h_errs(data):
    method_results = data["First Order Trotterization"]
    hs = list(method_results.keys())
    errors = list(method_results[hs[0]].keys())
    return hs, errors


from ising.hamiltonian import parametrized_ising


def main():
    json_name = f"data/benchmark/heat/{FILE_NAME}_{GATE}_h_eps.json"
    answers = _read_json(json_name)

    # Converting the logirithmic ranges to decimals
    h_points, error_points = get_h_errs(answers)

    best_method = {}
    best_gate_count = {}
    mx_gate, mn_gate = 0, float("inf")
    color_grid = []
    for h in h_points:
        best_method[h] = {}
        best_gate_count[h] = {}
        row = []

        for error in error_points:
            cur_min, cur_method = float("inf"), None
            for method in MAP.values():
                cur = answers[method][h][error]
                mx_gate = max(cur, mx_gate)
                mn_gate = min(cur, mn_gate)
                if cur < cur_min:
                    cur_min = cur
                    cur_method = method

            assert cur_method is not None
            best_method[h][error] = NUMBER[cur_method]
            best_gate_count[h][error] = cur_min
            row.append(NUMBER[cur_method])

        color_grid.append(row)

    cmap = sns.color_palette(COLORS)

    fig, ax = plt.subplots()
    sns.heatmap(color_grid, cmap=cmap, cbar=False, linewidths=0.5, linecolor="gray")

    plt.xlabel("Error")
    plt.ylabel("h")

    error_labels = [_truncate(x, 3) for x in error_points]
    h_labels = [_truncate(x, 3) for x in h_points]

    plt.xticks(np.arange(0.5, 10.5), error_labels, rotation=0)
    plt.yticks(np.arange(0.5, 10.5), h_labels, rotation=0)

    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=color,
            markersize=10,
            label=METHODS[i],
        )
        for i, color in enumerate(COLORS)
    ]
    fig.subplots_adjust(left=0.13, bottom=0.10, right=0.85, top=0.85)
    plt.legend(handles=legend_elements, loc=(-0.0, 1.04), ncol=2)

    diagram_name = f"plots/benchmark/grid/{FILE_NAME}_{GATE}_h_vs_eps.png"
    print(f"Saving diagram at:{diagram_name}")
    plt.savefig(diagram_name, dpi=300)


if __name__ == "__main__":
    main()
