import numpy as np
from ising.hamiltonian import Hamiltonian
from ising.groundstate.simulation.utils import (
    ground_state_constants,
    ground_state_maximum_time,
)
from ising.hamiltonian.ising_one import trotter_reps_general
from ising.utils import Decomposer
from qiskit.circuit.library import PauliEvolutionGate
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli

from ising.benchmark.gates.counter import Counter


def gates_for_pauli(
    decomposer: Decomposer, pauli: Pauli, time: float
) -> dict[str, int]:
    evo = PauliEvolutionGate(pauli, time=time)
    circuit = QuantumCircuit(evo.num_qubits)
    circuit.append(evo, range(evo.num_qubits))
    dqc = decomposer.decompose(circuit)
    return dict(dqc.count_ops())


class TrotterBenchmark:
    def __init__(self, ham: Hamiltonian, observable_norm: float, **kwargs):
        """
        Benchmark calculator for Trotterization based groundstate preparation.

        Inputs:
            - observable_norm: Norm of observable to run LCU simulation on.

        Kwargs must contain:
            - overlap
            - error
            - success
        """
        self.obs_norm = observable_norm

        # Useful constants for single-ancilla LCU groundstate
        self.overlap = kwargs["overlap"]
        self.error = kwargs["error"]
        self.success = kwargs["success"]

        self.ham = ham
        self.ground_params = ground_state_constants(
            self.ham._approx_spectral_gap,
            self.overlap,
            self.error,
            self.success,
            self.obs_norm,
        )

    def simulation_gate_count(self, time: float):
        """
        Calculates the gate depth for givin time
        """
        reps = trotter_reps_general(self.ham.sparse_repr, time, self.error)

        count = Counter()
        decomposer = Decomposer()
        for pauli, coeff in zip(
            self.ham.sparse_repr.paulis, self.ham.sparse_repr.coeffs
        ):
            gate_dict = gates_for_pauli(decomposer, pauli, coeff.real * time / reps)
            count.add(gate_dict)

        return count.times(reps)

    def calculate_gates(self):
        """
        Calculates the gate depth for GSP for given Hamiltonian
        """
        max_time = ground_state_maximum_time(self.ground_params)
        return self.simulation_gate_count(max_time)
