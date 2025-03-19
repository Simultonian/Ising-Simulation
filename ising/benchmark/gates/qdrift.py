from tqdm import tqdm
from typing import Union
import numpy as np
from qiskit.circuit.library import PauliEvolutionGate
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, SparsePauliOp
import sys

from ising.hamiltonian import Hamiltonian
from ising.groundstate.simulation.utils import (
    ground_state_constants,
    ground_state_maximum_time,
)
from ising.hamiltonian.ising_one import qdrift_count
from ising.utils import Decomposer
from ising.utils.gate_count import count_non_trivial

from ising.benchmark.gates.counter import Counter


def gates_for_pauli(
    decomposer: Decomposer, pauli: Union[Pauli, SparsePauliOp], time: float
) -> dict[str, int]:
    evo = PauliEvolutionGate(pauli, time=time)
    circuit = QuantumCircuit(evo.num_qubits)
    circuit.append(evo, range(evo.num_qubits))
    dqc = decomposer.decompose(circuit)
    return dict(dqc.count_ops())


class qDRIFTBenchmark:
    def __init__(self, ham: Hamiltonian, observable_norm: float, **kwargs):
        """
        Benchmark calculator for qDRIFT based groundstate preparation.

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

        self.optimise = kwargs.get("optimise", False)
        self.decomposer = Decomposer()

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
        print("Running qDRIFT Gate Count")
        lambd = np.sum(np.abs(self.ham.sparse_repr.coeffs))
        reps = qdrift_count(lambd, time, self.error)

        print(f"time:{time} lambda:{lambd} reps:{reps}")

        count = Counter()

        if not self.optimise:
            total_count = len(self.ham.sparse_repr.coeffs)
            with tqdm(total=total_count) as pbar:
                for pauli, coeff in zip(
                    self.ham.sparse_repr.paulis, self.ham.sparse_repr.coeffs
                ):
                    gate_dict = gates_for_pauli(
                        self.decomposer, pauli, lambd * time / reps
                    )
                    pbar.update(1)
                    count.weighted_add(abs(coeff.real), gate_dict)

            return count.times(reps)

        gate_dict = gates_for_pauli(self.decomposer, self.ham.sparse_repr, time / reps)
        count.add(gate_dict)
        return count.times(reps)

    def calculate_gates(self) -> dict[str, int]:
        """
        Calculates the gate depth for GSP for given Hamiltonian
        """
        max_time = ground_state_maximum_time(self.ground_params)
        return self.simulation_gate_count(max_time)


def qdrift_gates(
    ham: Hamiltonian, obs_norm: float, overlap: float, error: float, success: float
) -> dict[str, int]:
    benchmarker = qDRIFTBenchmark(
        ham, obs_norm, overlap=overlap, error=error, success=success
    )

    return benchmarker.calculate_gates()


SPLIT_SIZE = 100


class QDriftBenchmarkTime:
    def __init__(self, ham: Hamiltonian):
        """
        Benchmark calculator for Trotterization based groundstate preparation.

        Inputs:
            - ham: Hamitlonian
        """

        self.ham = ham
        self.decomposer = Decomposer()

        coeffs = []
        paulis = []
        sign_map = {}
        for op, coeff in self.ham.sparse_repr.to_list():
            if coeff.real < 0:
                sign_map[Pauli(op)] = -1
                coeff = -coeff
            else:
                sign_map[Pauli(op)] = 1

            coeffs.append(coeff.real)
            paulis.append(Pauli(op))

        self.coeffs = np.abs(np.array(coeffs))
        self.lambd = sum(self.coeffs)
        self.paulis = paulis
        self.sign_map = sign_map
        self.indices = list(range(len(self.paulis)))

    def simulation_circuit(self, time: float, reps: int) -> QuantumCircuit:
        """
        Calculates the gate depth for given time
        """
        evolution_time = self.lambd * time / reps
        print(f"QDrift:: evolution_time={evolution_time} lambda={self.lambd} time={time} reps={reps}")
        circuit = QuantumCircuit(self.ham.num_qubits)

        samples = np.random.choice(self.indices, p=self.coeffs / self.lambd, size=reps)
        # Could be heavy operation for large reps.
        for sample in samples:
            pauli = self.paulis[sample]
            evo = PauliEvolutionGate(pauli, time=evolution_time)
            circuit.append(evo, range(evo.num_qubits))

        return circuit

    def simulation_gate_count(self, time: float, reps: int) -> dict[str, int]:
        print(f"QDrift:: Running gate count for time={time} reps={reps}")
        if reps < SPLIT_SIZE:
            split = 1
        else:
            split = SPLIT_SIZE

        circuit = self.simulation_circuit(time, split)
        dqc = self.decomposer.decompose(circuit)

        counter = Counter()
        counter.add(dict(dqc.count_ops()))
        print(f"QDrift:: counter: {counter.times(1)} reps: {reps}")
        return counter.times(reps // split)

    def circuit_gate_count(self, gate: str, time: int) -> int:
        """
        Counts the gates analytically rather than via decomposition.
        """
        t_bar = time * self.lambd
        reps = max(20, int(10 * np.ceil(t_bar) ** 2))

        if gate == "cx":
            print(f"QDrift: Counting cx for reps:{reps}")
            total = 0
            # Each rep has only one Pauli exponentiation.
            samples = np.random.choice(
                self.indices, p=self.coeffs / self.lambd, size=reps
            )
            for sample in samples:
                pauli = self.paulis[sample]
                count = count_non_trivial(pauli)
                total += 2 * (count - 1)
            return total
        else:
            return 0

    def controlled_gate_count(self, time: float, reps: int) -> dict[str, int]:
        print(f"QDrift:: Running controlled gate count for time={time}")
        if reps < SPLIT_SIZE:
            split = 1
        else:
            split = SPLIT_SIZE

        big_circ = QuantumCircuit(self.ham.num_qubits + 1)
        controlled_gate = self.simulation_circuit(time, split).to_gate().control(1)
        big_circ.append(controlled_gate, range(self.ham.num_qubits + 1))
        dqc = self.decomposer.decompose(big_circ)

        counter = Counter()
        counter.add(dict(dqc.count_ops()))
        return counter.times(reps // split)


from ising.hamiltonian.ising_one import parametrized_ising


def main():
    num_qubits, h = 10, 0.125
    eps = 0.1
    time = 20
    hamiltonian = parametrized_ising(num_qubits, h)
    lambd = np.sum(np.abs(hamiltonian.coeffs))
    reps = qdrift_count(lambd, time, eps)

    print(f"reps:{reps}")

    benchmarker = QDriftBenchmarkTime(hamiltonian)

    print(benchmarker.simulation_gate_count(time, reps))
    print(benchmarker.controlled_gate_count(time, reps))


if __name__ == "__main__":
    main()
