from typing import Union
from functools import cache
import numpy as np
from tqdm import tqdm

from qiskit.circuit.library import PauliEvolutionGate
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, SparsePauliOp

from ising.hamiltonian import Hamiltonian
from ising.groundstate.simulation.utils import (
    ground_state_constants,
    ground_state_maximum_time,
)

from ising.benchmark.gates.counter import Counter
from ising.utils.gate_count import count_non_trivial
from ising.hamiltonian.ising_one import trotter_reps_general
from ising.utils import Decomposer
from ising.utils.pauligate import PauliCounter


def gates_for_pauli(
    decomposer: Decomposer, pauli: Union[Pauli, SparsePauliOp], time: float
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
        print("Running Trotter Gate Count")
        reps = trotter_reps_general(self.ham.sparse_repr, time, self.error)
        print(f"time:{time} reps:{reps}")

        count = Counter()
        if not self.optimise:
            total_count = len(self.ham.sparse_repr.coeffs)
            with tqdm(total=total_count) as pbar:
                for pauli, coeff in zip(
                    self.ham.sparse_repr.paulis, self.ham.sparse_repr.coeffs
                ):
                    gate_dict = gates_for_pauli(
                        self.decomposer, pauli, coeff.real * time / reps
                    )
                    pbar.update(1)
                    count.add(gate_dict)

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


def trotter_gates(
    ham: Hamiltonian, obs_norm: float, overlap: float, error: float, success: float
) -> dict[str, int]:
    benchmarker = TrotterBenchmark(
        ham, obs_norm, overlap=overlap, error=error, success=success
    )

    return benchmarker.calculate_gates()


SPLIT_SIZE = 1


class TrotterBenchmarkTime:
    def __init__(self, ham: Hamiltonian, system: str):
        """
        Benchmark calculator for Trotterization based groundstate preparation.

        Inputs:
            - ham: Hamitlonian
        """

        self.ham = ham
        if system == "":
            self.use_system = False
        else:
            self.use_system = True

        self.system = system
        self.decomposer = Decomposer()

    def simulation_circuit(self, time: float, reps: int) -> QuantumCircuit:
        """
        Calculates the gate depth for given time
        """
        circuit = QuantumCircuit(self.ham.num_qubits)

        # print("starting circuit")
        for _ in range(reps):
            for pauli, _coeff in zip(self.ham.paulis, self.ham.coeffs):
                coeff = abs(_coeff)
                evo = PauliEvolutionGate(pauli, time=coeff * time / reps)
                circuit.append(evo, range(evo.num_qubits))

        # print("circuit complete")
        return circuit

    @cache
    def controlled_simulation_circuit_decomposed(self, time: float, reps: int) -> Counter:
        """
        Get the simulation circuit when controlled
        """

        print("starting circuit")
        counter = Counter()

        if self.use_system:
            # Coeff remains same for the system
            pauli_counter = PauliCounter(self.system, self.ham.num_qubits, time / reps)
            prev_data = pauli_counter.control_data
        else:
            pauli_counter = None
            prev_data = {}

        if len(prev_data) != len(self.ham.paulis):
            data = {}
            with tqdm(total=len(self.ham.paulis)) as pbar:
                for pauli, _coeff in zip(self.ham.paulis, self.ham.coeffs):
                    coeff = abs(_coeff)
                    evo = PauliEvolutionGate(pauli, time=coeff * time / reps).control(1)

                    circuit = QuantumCircuit(evo.num_qubits)
                    circuit.append(evo, range(evo.num_qubits))
                    count = dict(self.decomposer.decompose(circuit).count_ops())
                    counter.add(count)
                    data[str(pauli)] = count

                    pbar.update(1)
            result = {"total": counter.count, "individual": data}
            if pauli_counter is not None:
                pauli_counter.set_control_data(result)
            print("circuit complete")
        else:
            print("Loaded from Pauli Counter")
            if pauli_counter is not None:
                counter.add(pauli_counter.control_total)
        
        new_counter = Counter()
        new_counter.weighted_add(reps, counter.count)
        return new_counter 

    def circuit_gate_count(self, gate: str, reps: int) -> int:
        """
        Counts the gates analytically rather than via decomposition.
        """
        if gate == "cx":
            print(f"Trotter: Counting cx for reps:{reps}")
            total = 0
            for pauli in self.ham.paulis:
                count = count_non_trivial(pauli)
                total += 2 * (count - 1)
            return total * reps
        else:
            return 0

    def simulation_gate_count(self, time: float, reps: int) -> dict[str, int]:
        print(f"Trotter: Running gate count for time: {time}")

        if reps < SPLIT_SIZE:
            split = 1
        else:
            split = SPLIT_SIZE

        circuit = self.simulation_circuit(time, split)
        # print("decomposing")
        dqc = self.decomposer.decompose(circuit)
        # print("decomposed")
        print(dqc.count_ops())

        counter = Counter()
        counter.add(dict(dqc.count_ops()))
        return counter.times(reps // split)

    def controlled_gate_count(self, time: float, reps: int) -> dict[str, int]:
        print(f"Trotter: Running controlled gate count for time: {time}")

        if reps < SPLIT_SIZE:
            split = 1
        else:
            split = SPLIT_SIZE

        counter = self.controlled_simulation_circuit_decomposed(time, split)

        return counter.times(reps // split)


from ising.hamiltonian.ising_one import parametrized_ising
from ising.hamiltonian.ising_one import trotter_reps


def main():
    num_qubits, h = 10, 0.125
    eps = 0.1
    time = 20
    reps = trotter_reps(num_qubits, h, time, eps)
    print(f"reps:{reps}")
    hamiltonian = parametrized_ising(num_qubits, h)
    benchmarker = TrotterBenchmarkTime(hamiltonian)
    print(benchmarker.simulation_gate_count(time, reps))
    print(benchmarker.controlled_gate_count(time, reps))


if __name__ == "__main__":
    main()
