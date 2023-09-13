from typing import Optional
from functools import lru_cache
import numpy as np
from numpy.typing import NDArray
from collections import Counter

from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import Operator
from qiskit.synthesis import LieTrotter, QDrift
from qiskit.circuit.library import PauliEvolutionGate
from ising.hamiltonian import Hamiltonian, qdrift_count
from ising.hamiltonian.hamiltonian import substitute_parameter
from ising.simulation.trotter import Lie
from ising.simulation.singlelcu.singlelcu import calculate_mu
