import hashlib
import json
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

"""
Quantum Collision Model Benchmarking Tool

This script generates analytical plots comparing gate count vs time for different
quantum simulation techniques for Lindbladian dynamics using the collision model.

To use: Simply adjust the parameters in the 'parameters' dictionary and run the script.
"""

# Directory to save plots - create if it doesn't exist
DIR = "plots/newbenchmark/cx_vs_collision_analytical/"
os.makedirs(DIR, exist_ok=True)

# Parameters - MODIFY THESE VALUES AS NEEDED
parameters = {
    "L": 10,                    # Number of Pauli strings
    "m": 10,                    # Number of sub-environments
    "O_norm": 9.0,              # Norm of observable O
    "H_norm": 1.0,              # Norm of Hamiltonian H
    "Gamma": 24.0,              # Gamma parameter
    "beta_max": 1.0,            # Maximum beta
    "epsilon": 1e-4,            # Fixed error value
    "alpha_commutator_1st": 1.1114244518026801,  # Alpha commutator for 1st-order Trotter
    "alpha_commutator_2nd": 2.3818703023137573,  # Alpha commutator for 2nd-order Trotter
    "delta_t": 0.001,
    "k_values": np.linspace(10, 1000, 500).tolist()  # Number of collisions
}

COLORS = ["#DC5B5A", "#625FE1", "#94E574", "#2A2A2A", "#D575EF", 
          "#3ABEFF", "#FFB743", "#00CC99", "#FF6B6B", "#845EC2", 
          "#F9F871", "#00C9A7", "#C34A36", "#4ECDC4", "#FF9671", 
          "#FFC75F", "#008080", "#D65DB1", "#4D8076", "#FF8066"]

def generate_plots():
    # Generate a hash for the parameters to track different runs
    param_hash = hashlib.md5(json.dumps(parameters, sort_keys=True).encode()).hexdigest()
    parameters["hash"] = param_hash

    # Save parameters to JSON for reproducibility
    with open(os.path.join(DIR, f"parameters_{param_hash}.json"), "w") as f:
        json.dump(parameters, f, indent=4)

    # Extract parameters for readability
    L = parameters["L"]
    O_norm = parameters["O_norm"]
    beta_max = parameters["beta_max"]
    epsilon = parameters["epsilon"]
    alpha_commutator_1st = parameters["alpha_commutator_1st"]
    alpha_commutator_2nd = parameters["alpha_commutator_2nd"]
    delta_t = parameters["delta_t"]
    k_values = parameters["k_values"]

    def trotter_1st_order(K):
        # Circuit depth in terms of K, replacing L with alpha_commutator_1st
        circuit_depth = alpha_commutator_1st * ((beta_max ** 2) * (K ** 2) * O_norm * (delta_t ** 2)) / epsilon
        return circuit_depth + 2 * K


    def qdrift(K):
        circuit_depth = ((beta_max**2) * (K ** 2) * O_norm * (delta_t ** 2)) / epsilon
        return circuit_depth + 2 * K

    def trotter_2nd_order(K):
        circuit_depth = alpha_commutator_2nd * (K * beta_max * delta_t)**(5/4) * (O_norm/epsilon)**(1/4)
        return circuit_depth + 2 * K

    def single_ancilla_lcu(K):
        circuit_depth = (beta_max**2) * (K ** 2) * (delta_t ** 2) * (np.log(beta_max * K * O_norm * delta_t / epsilon) / np.log(np.log(beta_max * K * O_norm * delta_t / epsilon)))
        return circuit_depth + 2 * K
    
    # Generate gate counts for each algorithm
    trotter_counts = [trotter_1st_order(k) for k in k_values]
    qdrift_counts = [qdrift(k) for k in k_values]
    trotter_2nd_counts = [trotter_2nd_order(k) for k in k_values]
    single_ancilla_counts = [single_ancilla_lcu(k) for k in k_values]

    # Find the maximum value of the lower algorithms to set the y-axis limit
    # lower_algos_max = max(max(trotter_2nd_counts), max(single_ancilla_counts))
    
    # Add padding to the y-axis upper limit
    # y_limit = lower_algos_max * 1.2

    # Function to create plots with or without legend
    def create_plot(with_legend=True):
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        
        # Plot 1st-order Trotter with clip_on=False to allow it to extend beyond the plot bounds
        ax.plot(k_values, trotter_counts, '-', color=COLORS[0], 
                label='1st-order Trotter')
        # ax.scatter(time_values, trotter_counts, color=COLORS[0], s=50)
        
        # Plot other algorithms with higher zorder to ensure they're visible
        ax.plot(k_values, qdrift_counts, '-', color=COLORS[1], 
                label='QDrift', zorder=2)
        # ax.scatter(time_values, qdrift_counts, color=COLORS[1], s=50, zorder=2)
        
        ax.plot(k_values, trotter_2nd_counts, '-', color=COLORS[2], 
                label='2nd-order Trotter', zorder=2)
        # ax.scatter(time_values, trotter_2nd_counts, color=COLORS[2], s=50, zorder=2)
        
        ax.plot(k_values, single_ancilla_counts, '-', color=COLORS[3], 
                label='Single-Ancilla LCU', zorder=2)
        # ax.scatter(time_values, single_ancilla_counts, color=COLORS[3], s=50, zorder=2)
        
        # Set the y-axis limit to focus on lower algorithms
        # plt.ylim(0, y_limit)
        
        # Add annotation to indicate 1st-order Trotter continues off-scale
        # last_visible_index = next((i for i, y in enumerate(trotter_counts) if y > y_limit), len(trotter_counts)) - 1
        # if last_visible_index >= 0:
        #     last_visible_x = time_values[last_visible_index]
        #     plt.annotate("1st-order Trotter\ncontinues off-scale", 
        #                 xy=(last_visible_x, y_limit * 0.95), 
        #                 xytext=(last_visible_x + 1, y_limit * 0.7),
        #                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
        #                 fontsize=9)
        
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.yscale('log')
        
        # Linear scale for y-axis as requested
        plt.xlabel(r'Collisions ($K$)')
        plt.ylabel(r'$\text{CNOT}$ Gate Count')
        
        if with_legend:
            plt.title('Gate Count vs Time (With Legend)')
            plt.legend()
        else:
            plt.title('')
        
        plt.tight_layout()
        filename = f"gate_count_vs_time_{'with' if with_legend else 'without'}_legend_{param_hash}.png"
        plt.savefig(os.path.join(DIR, filename), bbox_inches='tight')
        plt.close()

    # Create both plots
    create_plot(with_legend=True)
    create_plot(with_legend=False)

    print(f"Plots saved in {DIR} with hash {param_hash}")

if __name__ == "__main__":
    generate_plots()