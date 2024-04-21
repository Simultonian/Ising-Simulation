import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ising.hamiltonian import parametrized_ising
from ising.benchmark.gates.taylor import TaylorBenchmarkTime
from ising.benchmark.gates.trotter import TrotterBenchmarkTime
from ising.benchmark.gates.qdrift import QDriftBenchmarkTime
from ising.benchmark.gates.ktrotter import KTrotterBenchmarkTime
from ising.hamiltonian.ising_one import trotter_reps, qdrift_count
from ising.utils.commutator import commutator_r_second_order

def plot_gate_error(
    qubit, h_val, err_pair, point_count_pair, obs_norm, time_pair
):
    fig, ax = plt.subplots()

    # One col is fixed error
    error_points = [
        10**x for x in np.linspace(err_pair[0], err_pair[1], point_count_pair[0])
    ]
    # One row is fixed time
    time_points = [
        x for x in np.linspace(time_pair[0], time_pair[1], point_count_pair[1])
    ]

    # 2D Arrays where the first dim is time and second is error
    taylor, trotter, qdrift, ktrotter = [], [], [], []
    ham = parametrized_ising(qubit, h_val)
    lambd = np.sum(np.abs(ham.coeffs))

    taylor_bench = TaylorBenchmarkTime(ham)
    trotter_bench = TrotterBenchmarkTime(ham)
    qdrift_bench = QDriftBenchmarkTime(ham)
    ktrotter_bench = KTrotterBenchmarkTime(ham, order=2)

    nrows, ncols = len(time_points), len(error_points)
    for time in time_points:
        taylor.append([])
        trotter.append([])
        qdrift.append([])
        ktrotter.append([])

        for error in error_points:
            k = int(np.floor(np.log(lambd * time / error) / np.log(np.log(lambd * time / error))))

            taylor_counts = taylor_bench.simulation_gate_count(time, k)
            print(f"Taylor:{taylor_counts}")

            # Uses Ising Model specific bound
            trotter_rep = trotter_reps(qubit, h_val, time, error)
            trotter_counts = trotter_bench.simulation_gate_count(time, trotter_rep)
            print(f"Trotter:{trotter_counts}")

            qdrift_rep = qdrift_count(lambd, time, error)
            qdrift_counts = qdrift_bench.simulation_gate_count(time, qdrift_rep)
            print(f"qDRIFT:{qdrift_counts}")

            ktrotter_reps = commutator_r_second_order(ham.sparse_repr, time, error)
            ktrotter_counts = ktrotter_bench.simulation_gate_count(time, ktrotter_reps)
            print(f"kTrotter:{ktrotter_counts}")

            taylor[-1].append(np.log2(taylor_counts["cx"]))
            trotter[-1].append(np.log2(trotter_counts["cx"]))
            qdrift[-1].append(np.log2(qdrift_counts["cx"]))
            ktrotter[-1].append(np.log2(ktrotter_counts["cx"]))

            print("-----------------")

    taylor = np.array(taylor)
    trotter = np.array(trotter)
    qdrift = np.array(qdrift)
    ktrotter = np.array(ktrotter)

    max_lim = np.amax(taylor)
    max_lim = max(max_lim, np.amax(trotter))
    max_lim = max(max_lim, np.amax(qdrift))
    max_lim = max(max_lim, np.amax(ktrotter))

    min_lim = np.amin(taylor)
    min_lim = min(min_lim, np.amin(trotter))
    min_lim = min(min_lim, np.amin(qdrift))
    min_lim = min(min_lim, np.amin(ktrotter))

    mask = np.array(nrows * [ncols * [False, True, True, True]], dtype=bool)
    red = np.ma.masked_where(mask, np.repeat(qdrift, 4, axis=1))

    mask = np.array(nrows * [ncols * [True, False, True, True]], dtype=bool)
    blue = np.ma.masked_where(mask, np.repeat(trotter, 4, axis=1))

    mask = np.array(nrows * [ncols * [True, True, False, True]], dtype=bool)
    green = np.ma.masked_where(mask, np.repeat(taylor, 4, axis=1))

    mask = np.array(nrows * [ncols * [True, True, True, False]], dtype=bool)
    yellow = np.ma.masked_where(mask, np.repeat(ktrotter, 4, axis=1))


    # SETTING THE LIMITS AND PLOTTING
    redmesh = ax.pcolormesh(red, cmap='Blues', vmax=max_lim, vmin=min_lim)
    bluemesh = ax.pcolormesh(blue, cmap='Blues', vmax=max_lim, vmin=min_lim)
    greenmesh = ax.pcolormesh(green, cmap='Blues', vmax=max_lim, vmin=min_lim)
    yellowmesh = ax.pcolormesh(yellow, cmap='Blues', vmax=max_lim, vmin=min_lim)

    fig.subplots_adjust(left=0.08, bottom=0.05, right=0.86, top=0.88)

    # ADDING THE SIDE COLORBAR
    cbar = fig.colorbar(redmesh, cax=fig.add_axes([0.88, 0.05, 0.04, 0.83]))

    # TEXT ON THE COLORBAR
    cbar.ax.text(0.55, 0.22, r'CNOT Count ($\log_{10}$ scale)', rotation=90, ha='center', va='center',
                 transform=cbar.ax.transAxes, color='black')

    ax.set_ylabel(r"Time ($t$)")
    ax.yaxis.set_label_coords(-0.05, 0.51) 
    ax.set_xlabel(r"Error ($\log_{10}$ scale)")
    ax.xaxis.tick_top()
    ax.yaxis.tick_left()

    # SETTING THE TICK MARKS
    x_ticks = [x*4 for x in list(range(1, point_count[0] + 1))]
    ax.set_xticks(x_ticks)
    ax.set_yticks(list(range(point_count[1])))
    ax.tick_params(direction='out')

    # SETTING THE LABELS
    error_marks = [np.log10(x) for x in error_points]
    ax.set_xticklabels(['{}'.format(i) for i in error_marks])
    ax.set_yticklabels(['{}'.format(i) for i in time_points])


    # LABELLING the BARS
    ax.text(0.5, 0.35, 'qDRIFT', rotation=90, ha='center', va='center', color='black')
    ax.text(1.5, 1.09, 'First Order Trotterization', rotation=90, ha='center', va='center', color='black')
    ax.text(2.5, 1.05, 'Truncated Taylor Series', rotation=90, ha='center', va='center', color='black')
    ax.text(3.5, 0.85, 'kth Order Trotter', rotation=90, ha='center', va='center', color='black')

    for x_stamp in x_ticks:
        ax.axvline(x=x_stamp, color='black', linestyle='-')


    diagram_name = "plots/benchmark/heat/ising_cx_time.png"
    print(f"Saving diagram at:{diagram_name}")
    plt.savefig(diagram_name, dpi=300)


if __name__ == "__main__":
    qubit = 10
    h_val = 0.1
    err_pair = (-1, -3)
    # error, time
    point_count = (2, 5)
    obs_norm = 1
    time_pair = (1, 20)
    plot_gate_error(
        qubit, h_val, err_pair, point_count, obs_norm, time_pair
    )
