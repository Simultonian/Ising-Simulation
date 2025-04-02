import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import hashlib
import time

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
TIME = 1
ERROR_RANGE = (-1, -5)
ERROR_COUNT = 10


COLORS = ["#DC5B5A", "#625FE1", "#94E574", "#2A2A2A", "#D575EF", 
          "#3ABEFF", "#FFB743", "#00CC99", "#FF6B6B", "#845EC2", 
          "#F9F871", "#00C9A7", "#C34A36", "#4ECDC4", "#FF9671", 
          "#FFC75F", "#008080", "#D65DB1", "#4D8076", "#FF8066"]
DIR = "plots/newbenchmark/cx_vs_error_new/"


results = {
        "trotter": [
            158490000.0,
            1150920000.0,
            8577468000.0,
            64893342600.0,
            495456247800.0,
            3804383958900.0,
            29306366019000.0,
            226221316040400.0,
            1748365097420700.0,
            1.35221399778465e+16
        ],
        "ktrotter": [
            4800000.0,
            15846000.0,
            55728000.0,
            193860000.0,
            701298000.0,
            2452107000.0,
            8772435000.0,
            31384422000.0,
            113205015000.0,
            404999595000.0
        ],
        "qdrift": [
            6241200.0,
            40289984.0,
            279088920.0,
            1970169024.0,
            14305831848.0,
            104162035712.0,
            801528381440.0,
            6038466116000.0,
            46930432021088.0,
            365048274951360.0
        ],
        "taylor": [
            40000.0,
            177920.0,
            309600.0,
            861600.0,
            2397600.0,
            6672400.0,
            18566000.0,
            51661600.0,
            143752400.0,
            399999600.0
        ]
    }

def test_main():
    np.random.seed(100)

    os.makedirs(DIR, exist_ok=True)
    
    # Generate errors
    errors = [10 ** x for x in np.linspace(ERROR_RANGE[0], ERROR_RANGE[1], ERROR_COUNT)]
    
    # Lists to store gate counts
    trotter_counts = results["trotter"]
    ktrotter_counts = results["ktrotter"]
    qdrift_counts = results["qdrift"]
    taylor_counts = results["taylor"]
    
    # Collect data
    # for error in errors:
    #     result_array = new_gate_counts(
    #         system_size=QUBIT_COUNT, 
    #         h_val=H_VAL, 
    #         evolution_time=TIME, 
    #         precision=error, 
    #         gamma=GAMMA
    #     )
    #     trotter_cx, ktrotter_cx, qdrift_cx, taylor_cx = result_array[0], result_array[1], result_array[2], result_array[3]
        
    #     trotter_counts.append(trotter_cx)
    #     ktrotter_counts.append(ktrotter_cx)
    #     qdrift_counts.append(qdrift_cx)
    #     taylor_counts.append(taylor_cx)
    
    # Generate a unique hash for the plots
    timestamp = int(time.time())
    hash_input = f"{timestamp}_{QUBIT_COUNT}_{H_VAL}_{TIME}_{GAMMA}"
    plot_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
    
    # Create parameter dictionary for JSON
    parameters = {
        "plot_hash": plot_hash,
        "qubit_count": QUBIT_COUNT,
        "h_val": H_VAL,
        "evolution_time": TIME,
        "gamma": GAMMA,
        "error_range": ERROR_RANGE,
        "error_count": ERROR_COUNT,
        "timestamp": timestamp,
        "results": {
                "trotter": trotter_counts,
                "ktrotter": ktrotter_counts,
                "qdrift": qdrift_counts,
                "taylor": taylor_counts,
        }
    }
    
    # Update parameters.json
    json_path = os.path.join(DIR, "parameters.json")
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            all_params = json.load(f)
    else:
        all_params = {}
    
    all_params[plot_hash] = parameters
    
    with open(json_path, 'w') as f:
        json.dump(all_params, f, indent=4)
    
    # Create plots with overlapping line and scatter
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Line and scatter plots overlapping
    ax = sns.lineplot(x=errors, y=trotter_counts, label='Trotter', color=COLORS[0], ax=ax, linewidth=3)
    ax = sns.scatterplot(x=errors, y=trotter_counts, color=COLORS[0], s=60, ax=ax)
    
    ax = sns.lineplot(x=errors, y=ktrotter_counts, label='K-Trotter', color=COLORS[1], ax=ax, linewidth=3)
    ax = sns.scatterplot(x=errors, y=ktrotter_counts, color=COLORS[1], s=60, ax=ax)
    
    ax = sns.lineplot(x=errors, y=qdrift_counts, label='QDrift', color=COLORS[2], ax=ax, linewidth=3)
    ax = sns.scatterplot(x=errors, y=qdrift_counts, color=COLORS[2], s=60, ax=ax)
    
    ax = sns.lineplot(x=errors, y=taylor_counts, label='Taylor', color=COLORS[3], ax=ax, linewidth=3)
    ax = sns.scatterplot(x=errors, y=taylor_counts, color=COLORS[3], s=60, ax=ax)
    
    # Set log scales
    plt.xscale('log')
    plt.yscale('log')
    
    # Remove the top and right border
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Add labels
    # plt.xlabel('Error', fontsize=17)
    # plt.ylabel('CNOT gate counts per coherent run', fontsize=17)
    plt.title(f'CX Gate Count vs Error (n={QUBIT_COUNT}, h={H_VAL}, t={TIME}, γ={GAMMA})')
    # plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.gca().invert_xaxis()
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
    print(f"Parameters saved to: {json_path}")



if __name__ == "__main__":
    test_main()
