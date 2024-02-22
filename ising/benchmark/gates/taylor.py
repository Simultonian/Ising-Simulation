from typing import Union
from functools import cache
import numpy as np
from ising.hamiltonian import Hamiltonian
from ising.groundstate.simulation.utils import (
    ground_state_constants,
    ground_state_maximum_time,
)
from ising.utils import Decomposer
from qiskit.circuit.library import PauliEvolutionGate, PauliGate
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, SparsePauliOp, PauliList

from ising.benchmark.gates.counter import Counter

from tqdm import tqdm


def gates_for_pauli(
    decomposer: Decomposer, pauli: Union[Pauli, SparsePauliOp], time: float
) -> dict[str, int]:
    evo = PauliEvolutionGate(pauli, time=time)
    circuit = QuantumCircuit(evo.num_qubits)
    circuit.append(evo, range(evo.num_qubits))
    dqc = decomposer.decompose(circuit)
    return dict(dqc.count_ops())


def gates_for_pauli_direct(decomposer: Decomposer, pauli: Pauli) -> dict[str, int]:
    evo = PauliGate(str(pauli))
    circuit = QuantumCircuit(evo.num_qubits)
    circuit.append(evo, range(evo.num_qubits))
    dqc = decomposer.decompose(circuit)
    return dict(dqc.count_ops())


class TaylorBenchmark:
    def __init__(self, ham: Hamiltonian, observable_norm: float, **kwargs):
        """
        Benchmark calculator for Truncated Taylor Series
        based groundstate preparation.

        Inputs:
            - observable_norm: Norm of observable to run LCU simulation on.

        Kwargs must contain:
            - overlap
            - error
            - success
        """
        self.ham = ham
        self.obs_norm = observable_norm

        # Useful constants for single-ancilla LCU groundstate
        self.overlap = kwargs["overlap"]
        self.error = kwargs["error"]
        self.success = kwargs["success"]

        self.ground_params = ground_state_constants(
            self.ham._approx_spectral_gap,
            self.overlap,
            self.error,
            self.success,
            self.obs_norm,
        )

        self.optimise = kwargs.get("optimise", False)

        paulis, coeffs = [], []
        for pauli, _coeff in zip(self.ham.paulis, self.ham.coeffs):
            assert _coeff.imag == 0
            coeff = _coeff.real
            paulis.append(pauli)
            coeffs.append(abs(coeff))

        self.paulis, self.coeffs = PauliList(paulis), np.array(coeffs)
        self.beta = np.sum(np.array(self.coeffs))
        self.coeffs /= self.beta
        self.decomposer = Decomposer()

    @cache
    def average_pauli_depth(self) -> dict[str, int]:
        """
        Calculates the average depth of just constructing the Pauli gate
        """
        count = Counter()
        total_count = len(self.ham.sparse_repr.coeffs)
        with tqdm(total=total_count) as pbar:
            for pauli, coeff in zip(self.paulis, self.coeffs):
                gate_dict = gates_for_pauli_direct(self.decomposer, pauli)
                count.weighted_add(coeff.real, gate_dict)
                pbar.update(1)

        return count.times(1)

    def average_pauli_rotate_depth(self, time) -> dict[str, int]:
        """
        Calculates the average depth of constructing rotation Pauli gate
        """
        print("Average rotation depth")
        count = Counter()
        total_count = len(self.ham.sparse_repr.coeffs)
        with tqdm(total=total_count) as pbar:
            for pauli, coeff in zip(self.paulis, self.coeffs):
                gate_dict = gates_for_pauli(self.decomposer, pauli, time)
                count.weighted_add(coeff.real, gate_dict)
                pbar.update(1)

        return count.times(1)

    def error_gate_count(self, err: float) -> dict[str, int]:
        """
        Calculates the gate depth for given error
        """
        self.ground_params = ground_state_constants(
            self.ham._approx_spectral_gap,
            self.overlap,
            err,
            self.success,
            self.obs_norm,
        )
        return self.calculate_gates()

    def simulation_gate_count(self, time: float) -> dict[str, int]:
        """
        Calculates the gate depth for givin time
        """
        # r = (beta t) ^ 2
        reps = (self.beta * time) ** 2
        k = np.floor(np.log(self.beta * time / self.error) / np.log(np.log(self.beta * time / self.error)))
        print(f"Running TTS Gate Count, reps:{reps} k:{k}")

        count = Counter()
        pauli_depth = self.average_pauli_depth()
        count.weighted_add(k, pauli_depth)
        rotation_depth = self.average_pauli_rotate_depth(time)
        count.add(rotation_depth)

        return count.times(reps)

    def calculate_gates(self) -> dict[str, int]:
        """
        Calculates the gate depth for GSP for given Hamiltonian
        """
        max_time = ground_state_maximum_time(self.ground_params)
        return self.simulation_gate_count(max_time)


def taylor_gates(
    ham: Hamiltonian, obs_norm: float, overlap: float, error: float, success: float
) -> dict[str, int]:
    benchmarker = TaylorBenchmark(
        ham, obs_norm, overlap=overlap, error=error, success=success
    )

    return benchmarker.calculate_gates()
