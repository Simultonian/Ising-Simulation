# Ising-Simulation
Functionality for testing Ising model simulation using different methods

This repository contains several simulations, primarily focusing on the comparitive study of CNOT gate count for several Hamiltonian simulation techniques. The Hamiltonian simulation technique is applied to various problems, such as ground-state preparation, Lindbladian dynamics, and collision model simulations. The data is stored in several forms including a database as well as JSON. Most recently, we have moved to a database format to minimize the overhead of file management and data loading. The data has been illustrated in `ising/plots` holding the same directory structure as the code used to generate it present in `ising/`.

## Collision Model Simulation
For analytical collision model code, please refer to `ising/newbenchmark` which contain self-contained scripts for performing various benchmarks. This folder also contains the scripts for running numerical benchmarking that generate the corresponding numerical numbers with the support of Qiskit library.

The analytical benchmarking scripts do not require the installation of Qiskit, while the numerical benchmarking scripts require them. The results presented by both these scripts have been noted to be significantly similar. If one wishes to quickly benchmark the Hamiltonian methods for the given parameters, they may choose to run the analytical scripts directly and generate the plots. The corresponding data to these plots is directly stored in `ising/function.db` via SQLite, the user may add some print statements to see the data in JSON format rather than the visual plot format.
