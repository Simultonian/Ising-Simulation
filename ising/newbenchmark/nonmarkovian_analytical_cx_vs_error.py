import hashlib
import json
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1.inset_locator as mpl_il


"""
Quantum Collision Model Benchmarking Tool

This script generates analytical plots comparing gate count vs time for different
quantum simulation techniques for Lindbladian dynamics using the collision model.

To use: Simply adjust the parameters in the 'parameters' dictionary and run the script.
"""

# Directory to save plots - create if it doesn't exist
DIR = "plots/newbenchmark/nonmarkovian_cx_vs_error_analytical/"
os.makedirs(DIR, exist_ok=True)

# Parameters - MODIFY THESE VALUES AS NEEDED
parameters = {
    "L": 10,                    # Number of Pauli strings
    "m": 10,                    # Number of sub-environments
    "O_norm": 9.0,              # Norm of observable O
    "H_norm": 1.0,              # Norm of Hamiltonian H
    "Gamma": 24.0,              # Gamma parameter
    "beta_max": 1.0,            # Maximum beta
    "K": 1e4,                   # Fixed number of collisions
    "alpha_commutator_1st": 1.1114244518026801,  # Alpha commutator for 1st-order Trotter
    "alpha_commutator_2nd": 2.3818703023137573,  # Alpha commutator for 2nd-order Trotter
    "delta_t": 0.001,
    "error_values": [10 ** x for x in np.linspace(-3, -6, 10).tolist()]  # Required precision
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
    K = parameters["K"]
    alpha_commutator_1st = parameters["alpha_commutator_1st"]
    alpha_commutator_2nd = parameters["alpha_commutator_2nd"]
    delta_t = parameters["delta_t"]
    error_values = parameters["error_values"]

    def trotter_1st_order(epsilon):
        # Circuit depth in terms of K, replacing L with alpha_commutator_1st
        circuit_depth = alpha_commutator_1st * ((beta_max ** 2) * (K ** 2) * O_norm * (delta_t ** 2)) / epsilon
        return circuit_depth + 2 * K

    def qdrift(epsilon):
        circuit_depth = ((beta_max**2) * (K ** 2) * O_norm * (delta_t ** 2)) / epsilon
        return circuit_depth + 2 * K

    def trotter_2nd_order(epsilon):
        circuit_depth = alpha_commutator_2nd * (K * beta_max * delta_t)**(5/4) * (O_norm/epsilon)**(1/4)
        return circuit_depth + 2 * K

    def single_ancilla_lcu(epsilon):
        circuit_depth = (beta_max**2) * (K ** 2) * (delta_t ** 2) * (abs(np.log(beta_max * K * O_norm * delta_t / epsilon)) / np.log(abs(np.log(beta_max * K * O_norm * delta_t / epsilon))))
        return circuit_depth + 2 * K
    
    # Generate gate counts for each algorithm
    trotter_counts = [trotter_1st_order(e) for e in error_values]
    qdrift_counts = [qdrift(e) for e in error_values]
    trotter_2nd_counts = [trotter_2nd_order(e) for e in error_values]
    single_ancilla_counts = [single_ancilla_lcu(e) for e in error_values]

    # Find the maximum value of the lower algorithms to set the y-axis limit
    # lower_algos_max = max(max(trotter_2nd_counts), max(single_ancilla_counts))
    
    # Add padding to the y-axis upper limit
    # y_limit = lower_algos_max * 1.2

    # Function to create plots with or without legend
    def create_plot(with_legend=True):
        plt.figure(figsize=(10, 6))
        ax = plt.gca()

        # Create inset axes in top-left corner (40% width, 30% height of main plot)
        ax_inset = mpl_il.inset_axes(ax, 
                           width="50%", 
                           height="40%",
                           bbox_to_anchor=(0.1, 0.1, 1, 1),  # Fine-tune this tuple to adjust position
                           bbox_transform=ax.transAxes,
                           loc=2,
                           borderpad=0)

        # Main plot: Single-Ancilla and 2nd-order Trotter
        ax.plot(error_values, single_ancilla_counts, '-', color=COLORS[3],
                label='Single-Ancilla LCU', zorder=2)
        ax.plot(error_values, trotter_2nd_counts, '-', color=COLORS[2],
                label='2nd-order Trotter', zorder=2)
        ax.scatter(error_values, single_ancilla_counts, s=25, color=COLORS[3],
                zorder=2)
        ax.scatter(error_values, trotter_2nd_counts, s=25, color=COLORS[2],
                zorder=2)

        # Inset plot: 1st-order Trotter and QDrift
        ax_inset.plot(error_values, trotter_counts, '-', color=COLORS[0],
                    label='1st-order Trotter')
        ax_inset.plot(error_values, qdrift_counts, '-', color=COLORS[1],
                    label='QDrift', zorder=2)
        ax_inset.scatter(error_values, trotter_counts, s=25, color=COLORS[0])
        ax_inset.scatter(error_values, qdrift_counts, s=25, color=COLORS[1])

        # Shared formatting
        for axis in [ax, ax_inset]:
            axis.set_yscale('log')
            axis.set_xscale('log')
            axis.invert_xaxis()
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)

        # Inset-specific adjustments
        ax_inset.tick_params(axis='both', labelsize=8)

        # Linear scale for y-axis as requested
        ax.set_xlabel(r'Error ($\epsilon$)')
        ax.set_ylabel(r'$\text{CNOT}$ Gate Count')
        
        if with_legend:
            # Main plot legend
            ax.legend(loc='upper right')
            ax_inset.legend(fontsize=8, frameon=False)
            plt.title('Gate Count vs Time (With Legend)')
            # plt.legend()
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