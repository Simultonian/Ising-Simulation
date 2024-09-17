import numpy as np
import matplotlib.pyplot as plt

def calculate_eta(n, g, G, phi, eta_values):
    if n == 0:
        return 1.0  # Initial condition
    
    sum_term = 0
    for j in range(n-1):
        sum_term += eta_values[j] * (-1j * np.exp(1j*phi) * np.cos(g) * np.sin(G))**(n-1-j)
    
    return np.cos(g) * (eta_values[-1] - np.tan(g)**2 * sum_term)

def main():
    # Parameters
    delta_tau = 0.05
    tau_max = 20
    lambda_param = 0.7  # Assuming λ = 1 for simplicity

    # Calculate G, phi, and g based on equations (13), (14), and (15)
    G = np.arcsin(np.exp(-delta_tau))
    phi = np.pi/2 - lambda_param * delta_tau
    g = np.sqrt(lambda_param/2 * delta_tau)

    # Calculate η_n values
    n_max = int(tau_max / delta_tau)
    eta_values = [calculate_eta(0, g, G, phi, [])]
    
    for n in range(1, n_max + 1):
        eta_n = calculate_eta(n, g, G, phi, eta_values)
        eta_values.append(eta_n)

    # Convert to numpy array for easier manipulation
    eta_array = np.array(eta_values)

    # Calculate |η_τ|^2
    eta_squared = np.abs(eta_array)**2

    # Create τ values
    tau_values = np.arange(0, tau_max + delta_tau, delta_tau)

    # Plot |η_τ|^2 against τ
    plt.figure(figsize=(10, 6))
    plt.plot(tau_values, eta_squared)
    plt.xlabel('τ')
    plt.ylabel('|η_τ|^2')
    plt.title('Plot of |η_τ|^2 against τ')
    plt.grid(True)
    file_name = f"plots/nonmarkovian/eeta.png"
    plt.savefig(file_name, dpi=300)
    print(f"Saving plot at {file_name}")

if __name__ == "__main__":
    main()
