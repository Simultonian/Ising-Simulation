import hashlib
import json
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

"""
Quantum Collision Model Benchmarking Tool

This script generates analytical plots comparing gate count vs error for different
quantum simulation techniques for Lindbladian dynamics using the collision model.

To use: Simply adjust the parameters in the 'parameters' dictionary and run the script.
"""

# Directory to save plots - create if it doesn't exist
DIR = "plots/newbenchmark/cx_vs_error_analytical/"
os.makedirs(DIR, exist_ok=True)

# Parameters - MODIFY THESE VALUES AS NEEDED
parameters = {
    "L": 10,                    # Number of Pauli strings
    "m": 10,                     # Number of sub-environments
    "O_norm": 9.0,              # Norm of observable O
    "H_norm": 1.0,              # Norm of Hamiltonian H
    "Gamma": 24.0,               # Gamma parameter
    "beta_max": 1.0,            # Maximum beta
    "alpha_commutator_1st": 1.1114244518026801,  # Alpha commutator for 1st-order Trotter
    "alpha_commutator_2nd": 2.3818703023137573,  # Alpha commutator for 2nd-order Trotter
    "alpha_commutator_2k": 1.0,   # Alpha commutator for 2k-order Trotter
    "epsilon_values": np.logspace(-1, -6, 10).tolist()  # Error values
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
    m = parameters["m"]
    O_norm = parameters["O_norm"]
    Gamma = parameters["Gamma"]
    beta_max = parameters["beta_max"]
    alpha_commutator_1st = parameters["alpha_commutator_1st"]
    alpha_commutator_2nd = parameters["alpha_commutator_2nd"]
    epsilon_values = parameters["epsilon_values"]

    # Analytical formulas for gate counts based on the Lindbladian benchmarking table
    def trotter_1st_order(epsilon):
        return alpha_commutator_1st * (L * m**3 * O_norm**2 * Gamma * beta_max**2) / epsilon**2

    def qdrift(epsilon):
        return (m**3 * O_norm**2 * Gamma * beta_max**2) / epsilon**2

    def trotter_2nd_order(epsilon):
        return alpha_commutator_2nd * L * (m)**(9/4) * (O_norm/epsilon)**(5/4) * Gamma * beta_max**(3/2)

    def single_ancilla_lcu(epsilon):
        return (m**3 * O_norm * Gamma * beta_max**2) / epsilon

    # Generate gate counts for each algorithm
    trotter_counts = [trotter_1st_order(eps) for eps in epsilon_values]
    qdrift_counts = [qdrift(eps) for eps in epsilon_values]
    trotter_2nd_counts = [trotter_2nd_order(eps) for eps in epsilon_values]
    single_ancilla_counts = [single_ancilla_lcu(eps) for eps in epsilon_values]

    # Plot with legend
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    ax = sns.lineplot(x=epsilon_values, y=trotter_counts, label='1st-order Trotter', color=COLORS[0], ax=ax)
    ax = sns.scatterplot(x=epsilon_values, y=trotter_counts, color=COLORS[0], s=50, ax=ax)
    
    ax = sns.lineplot(x=epsilon_values, y=qdrift_counts, label='QDrift', color=COLORS[1], ax=ax)
    ax = sns.scatterplot(x=epsilon_values, y=qdrift_counts, color=COLORS[1], s=50, ax=ax)
    
    ax = sns.lineplot(x=epsilon_values, y=trotter_2nd_counts, label='2nd-order Trotter', color=COLORS[2], ax=ax)
    ax = sns.scatterplot(x=epsilon_values, y=trotter_2nd_counts, color=COLORS[2], s=50, ax=ax)
    
    ax = sns.lineplot(x=epsilon_values, y=single_ancilla_counts, label='Single-Ancilla LCU', color=COLORS[3], ax=ax)
    ax = sns.scatterplot(x=epsilon_values, y=single_ancilla_counts, color=COLORS[3], s=50, ax=ax)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.gca().invert_xaxis()
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'Error ($\epsilon$)')
    plt.ylabel(r'$\text{CNOT}$ Gate Count')
    plt.title('Gate Count vs Error (With Legend)')
    plt.tight_layout()
    plt.savefig(os.path.join(DIR, f"gate_count_vs_error_with_legend_{param_hash}.png"))
    plt.close()

    # Plot without legend
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    ax = sns.lineplot(x=epsilon_values, y=trotter_counts, color=COLORS[0], ax=ax)
    ax = sns.scatterplot(x=epsilon_values, y=trotter_counts, color=COLORS[0], s=50, ax=ax)
    
    ax = sns.lineplot(x=epsilon_values, y=qdrift_counts, color=COLORS[1], ax=ax)
    ax = sns.scatterplot(x=epsilon_values, y=qdrift_counts, color=COLORS[1], s=50, ax=ax)
    
    ax = sns.lineplot(x=epsilon_values, y=trotter_2nd_counts, color=COLORS[2], ax=ax)
    ax = sns.scatterplot(x=epsilon_values, y=trotter_2nd_counts, color=COLORS[2], s=50, ax=ax)
    
    ax = sns.lineplot(x=epsilon_values, y=single_ancilla_counts, color=COLORS[3], ax=ax)
    ax = sns.scatterplot(x=epsilon_values, y=single_ancilla_counts, color=COLORS[3], s=50, ax=ax)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.title('')
    plt.gca().invert_xaxis()
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'Error ($\epsilon$)')
    plt.ylabel(r'$\text{CNOT}$ Gate Count')
    plt.title('')
    plt.tight_layout()
    plt.savefig(os.path.join(DIR, f"gate_count_vs_error_without_legend_{param_hash}.png"))
    plt.close()

    print(f"Plots saved in {DIR} with hash {param_hash}")

if __name__ == "__main__":
    generate_plots()