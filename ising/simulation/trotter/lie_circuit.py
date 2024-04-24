from typing import Optional, Union, Sequence
from functools import lru_cache
import numpy as np
from numpy.typing import NDArray

from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.synthesis import LieTrotter

from ising.hamiltonian import Hamiltonian, trotter_reps, general_grouping
from ising.hamiltonian.hamiltonian import substitute_parameter

from ising.hamiltonian.ising_one import parametrized_ising


def main():
    num_qubits, h = 10, 0.125
    hamiltonian = parametrized_ising(num_qubits, h)


if __name__ == "__main__":
    main()
