import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import hashlib
import time as sys_time

from qiskit.quantum_info import SparsePauliOp
from ising.hamiltonian import Hamiltonian, parametrized_ising_power, parametrized_ising
from ising.benchmark.gates import (
    TaylorBenchmarkTime,
    TrotterBenchmarkTime,
    QDriftBenchmarkTime,
    KTrotterBenchmarkTime,
)
from ising.hamiltonian.ising_one import qdrift_count
from ising.utils.commutator.commutator_hueristic import (
    r_first_order,
    r_second_order,
    alpha_commutator_second_order,
    alpha_commutator_first_order,
)
import json
from ising.lindbladian.simulation.utils import load_interaction_hams
from ising.utils import hache


QUBIT_COUNT = 10
GAMMA = 1.0
H_VAL = -0.1
ERROR = 0.01
TIME_RANGE = (1, 10)
TIME_COUNT = 10


COLORS = ["#DC5B5A", "#625FE1", "#94E574", "#2A2A2A", "#D575EF", 
          "#3ABEFF", "#FFB743", "#00CC99", "#FF6B6B", "#845EC2", 
          "#F9F871", "#00C9A7", "#C34A36", "#4ECDC4", "#FF9671", 
          "#FFC75F", "#008080", "#D65DB1", "#4D8076", "#FF8066"]
DIR = "plots/newbenchmark/cx_vs_time/"


@hache(blob_type=np.ndarray, max_size=1000)
def new_gate_counts(system_size, h_val, evolution_time, precision, gamma):
    nu = max(1, int(10 * (evolution_time ** 2) / precision))
    print(f"error={precision} nu={nu}")
    delta_t = evolution_time / nu

    ham_sys = parametrized_ising(system_size, h_val).sparse_repr
    big_ham_sys = SparsePauliOp([x.to_label() + "I" for x in ham_sys.paulis], ham_sys.coeffs)

    ham_ints_sparse = load_interaction_hams(system_size, gamma)

    taylors, trotters, ktrotters, qdrifts = [], [], [], []
    alpha_com_firsts = []
    alpha_com_seconds = []
    lambds = []

    ham_sim_error = precision / (9 * system_size * nu) # 9 is the observable norm, rest is number of collisions

    for ham_int_sparse in ham_ints_sparse:
        ham_sparse = (delta_t * big_ham_sys / system_size) + 1/np.sqrt(delta_t) * ham_int_sparse
        sorted_pairs = list(
            sorted(
                [(x, y.real) for (x, y) in zip(ham_sparse.paulis, ham_sparse.coeffs)],
                key=lambda x: abs(x[1]),
                reverse=True,
            )
        )
        alpha_com_first = alpha_commutator_first_order(
            sorted_pairs, ham_sim_error, delta=0, cutoff_count = -1
        )
        alpha_com_firsts.append(alpha_com_first)

        alpha_com_second = alpha_commutator_second_order(
            sorted_pairs, ham_sim_error, delta=0, cutoff_count = -1
        )
        alpha_com_seconds.append(alpha_com_second)

        # print(f"Commutators:: error={precision} first={alpha_com_first} second={alpha_com_second}")


        lambds.append(np.sum(np.abs(ham_sparse.coeffs)))

        taylors.append(TaylorBenchmarkTime(Hamiltonian(sparse_repr=ham_sparse)))
        trotters.append(TrotterBenchmarkTime(Hamiltonian(sparse_repr=ham_sparse), system=""))
        qdrifts.append(QDriftBenchmarkTime(Hamiltonian(sparse_repr=ham_sparse)))
        ktrotters.append(KTrotterBenchmarkTime(Hamiltonian(sparse_repr=ham_sparse), system="", order=2))

    ham_evo_time = delta_t

    trotter_cx, ktrotter_cx, qdrift_cx, taylor_cx = 0, 0, 0, 0
    for trotter, alpha_com_first in zip(trotters, alpha_com_firsts):
        # If alpha_com is given then sorted_pairs are not needed
        trotter_rep = r_first_order(
            sorted_pairs=[], time=ham_evo_time, error=ham_sim_error, alpha_com=alpha_com_first
        )
        count_trotter = trotter.simulation_gate_count(ham_evo_time, trotter_rep).get("cx", 0)
        # print(f"Trotter:: evo_time={ham_evo_time} rep={trotter_rep} sim_error={ham_sim_error} lambda={alpha_com_first} count={count_trotter}")
        trotter_cx += nu * count_trotter

    for ktrotter, alpha_com_second in zip(ktrotters, alpha_com_seconds):
        # If alpha_com is given then sorted_pairs are not needed
        ktrotter_rep = r_second_order(
            sorted_pairs=[], time=ham_evo_time, error=ham_sim_error, alpha_com=alpha_com_second
        )
        count_ktrotter = ktrotter.simulation_gate_count(ham_evo_time, ktrotter_rep).get("cx", 0)
        print(f"KTrotter:: evo_time={ham_evo_time} rep={ktrotter_rep} sim_error={ham_sim_error} lambda={alpha_com_second} count={count_ktrotter}")
        ktrotter_cx += nu * count_ktrotter

    for taylor, lambd in zip(taylors, lambds):
        k = int(
            np.floor(
                np.log(lambd * ham_evo_time / ham_sim_error) / np.log(np.log(lambd * ham_evo_time / ham_sim_error))
            )
        )
        taylor.capital_K = np.sqrt(nu * system_size)
        count_taylor = taylor.simulation_gate_count(ham_evo_time, k).get("cx", 0)
        print(f"Taylor:: evo_time={ham_evo_time} k={k} sim_error={ham_sim_error} lambda={lambd} count={count_taylor}")
        taylor_cx += nu * count_taylor

    for qdrift, lambd in zip(qdrifts, lambds):
        qdrift_rep = qdrift_count(lambd, ham_evo_time, ham_sim_error)
        qdrift_count_val = qdrift.controlled_gate_count(ham_evo_time, qdrift_rep).get("cx", 0)
        # print(f"QDrift:: evo_time={ham_evo_time} rep={qdrift_rep} sim_error={ham_sim_error} lambda={lambd} count={qdrift_count_val}")
        qdrift_cx += nu * qdrift_count_val

    return np.array([trotter_cx, ktrotter_cx, qdrift_cx, taylor_cx])


def test_main():
    np.random.seed(100)

    os.makedirs(DIR, exist_ok=True)
    
    # Generate errors
    times = [x for x in np.linspace(TIME_RANGE[0], TIME_RANGE[1], TIME_COUNT)]
    
    # Lists to store gate counts
    trotter_counts = []
    ktrotter_counts = []
    qdrift_counts = []
    taylor_counts = []
    
    # Collect data
    for time in times:
        result_array = new_gate_counts(
            system_size=QUBIT_COUNT, 
            h_val=H_VAL, 
            evolution_time=time, 
            precision=ERROR, 
            gamma=GAMMA
        )
        trotter_cx, ktrotter_cx, qdrift_cx, taylor_cx = result_array[0], result_array[1], result_array[2], result_array[3]
        
        trotter_counts.append(trotter_cx)
        ktrotter_counts.append(ktrotter_cx)
        qdrift_counts.append(qdrift_cx)
        taylor_counts.append(taylor_cx)
    
    # Generate a unique hash for the plots
    timestamp = int(sys_time.time())
    hash_input = f"{timestamp}_{QUBIT_COUNT}_{H_VAL}_{ERROR}_{GAMMA}"
    plot_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
    
    # Create parameter dictionary for JSON
    # parameters = {
    #     "plot_hash": plot_hash,
    #     "qubit_count": QUBIT_COUNT,
    #     "h_val": H_VAL,
    #     "error": ERROR,
    #     "gamma": GAMMA,
    #     "time_range": TIME_RANGE,
    #     "time_count": TIME_COUNT,
    #     "timestamp": timestamp,
    #     "results": {
    #             "trotter": trotter_counts,
    #             "ktrotter": ktrotter_counts,
    #             "qdrift": qdrift_counts,
    #             "taylor": taylor_counts,
    #     }
    # }
    
    # # Update parameters.json
    # json_path = os.path.join(DIR, "parameters.json")
    # if os.path.exists(json_path):
    #     with open(json_path, 'r') as f:
    #         all_params = json.load(f)
    # else:
    #     all_params = {}
    
    # all_params[plot_hash] = parameters
    
    # with open(json_path, 'w') as f:
    #     json.dump(all_params, f, indent=4)
    
    # Create plots with overlapping line and scatter
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Line and scatter plots overlapping
    ax = sns.lineplot(x=times, y=trotter_counts, label='Trotter', color=COLORS[0], ax=ax, linewidth=3)
    ax = sns.scatterplot(x=times, y=trotter_counts, color=COLORS[0], s=60, ax=ax)
    
    ax = sns.lineplot(x=times, y=ktrotter_counts, label='K-Trotter', color=COLORS[1], ax=ax, linewidth=3)
    ax = sns.scatterplot(x=times, y=ktrotter_counts, color=COLORS[1], s=60, ax=ax)
    
    ax = sns.lineplot(x=times, y=qdrift_counts, label='QDrift', color=COLORS[2], ax=ax, linewidth=3)
    ax = sns.scatterplot(x=times, y=qdrift_counts, color=COLORS[2], s=60, ax=ax)
    
    ax = sns.lineplot(x=times, y=taylor_counts, label='Taylor', color=COLORS[3], ax=ax, linewidth=3)
    ax = sns.scatterplot(x=times, y=taylor_counts, color=COLORS[3], s=60, ax=ax)
    
    # Set log scales
    # plt.xscale('log')
    plt.yscale('log')
    
    # Remove the top and right border
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Add labels
    # plt.xlabel('Evolution time', fontsize=14)
    # plt.ylabel('CNOT gate counts', fontsize=14)
    plt.title(f'CX Gate Count vs Error (n={QUBIT_COUNT}, h={H_VAL}, error={ERROR}, γ={GAMMA})')
    # plt.grid(True, which="both", ls="--", alpha=0.3)
    # plt.gca().invert_xaxis()
    plt.tight_layout(pad=4.0)
    plt.tick_params(axis='both', which='major', labelsize=28)

    
    # Save with legend
    plt.legend()
    plt.savefig(os.path.join(DIR, f"{plot_hash}_with_legend.png"), dpi=300)
    
    # Save without legend
    plt.legend().remove()
    plt.title("")
    plt.savefig(os.path.join(DIR, f"{plot_hash}_no_legend.png"), dpi=300)
    
    print(f"Plots saved with hash: {plot_hash}")
    # print(f"Parameters saved to: {json_path}")

if __name__ == "__main__":
    test_main()
