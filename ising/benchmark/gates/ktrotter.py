from typing import Union
import numpy as np
from ising.hamiltonian import Hamiltonian
from ising.groundstate.simulation.utils import (
    ground_state_constants,
    ground_state_maximum_time,
)
from ising.utils.gate_count import count_non_trivial

from ising.hamiltonian.ising_one import trotter_reps_general
from ising.utils import Decomposer
from qiskit.circuit.library import PauliEvolutionGate
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, SparsePauliOp

from ising.benchmark.gates.counter import Counter
from qiskit.synthesis import SuzukiTrotter

SPLIT_SIZE = 100


class KTrotterBenchmarkTime:
    def __init__(self, ham: Hamiltonian, order: int = 1):
        """
        Benchmark calculator for Trotterization based groundstate preparation.

        Inputs:
            - ham: Hamitlonian
        """

        self.ham = ham
        self.decomposer = Decomposer()
        self.order = order
        self.synth = SuzukiTrotter(order=order)

    def simulation_circuit(self, time: float, reps: int) -> QuantumCircuit:
        """
        Calculates the gate depth for given time
        """
        circuit = QuantumCircuit(self.ham.num_qubits)

        print("creating circuit")
        evo = PauliEvolutionGate(
            self.ham.sparse_repr, time=time / reps, synthesis=self.synth
        )
        circuit.append(evo, range(evo.num_qubits))

        # Could be heavy operation for large reps.
        print("created circuit")
        circuit = circuit.repeat(reps)
        return circuit

    def circuit_gate_count(self, gate: str, reps: int) -> int:
        """
        Counts the gates analytically rather than via decomposition.
        """
        if gate == "cx":
            print(f"Trotter: Counting cx for reps:{reps}")
            total = 0
            # Moving ahead with terms
            for pauli in self.ham.paulis:
                count = count_non_trivial(pauli)
                total += 2 * (count - 1)

            # Moving backwards with terms
            for pauli in self.ham.paulis[::-1]:
                count = count_non_trivial(pauli)
                total += 2 * (count - 1)

            return total * reps
        else:
            return 1

    def simulation_gate_count(self, time: float, reps: int) -> dict[str, int]:
        print(f"KTrotter: Running gate count for time: {time}")

        if reps < SPLIT_SIZE:
            split = 1
        else:
            split = SPLIT_SIZE

        circuit = self.simulation_circuit(time, split)
        dqc = self.decomposer.decompose(circuit)

        counter = Counter()
        counter.add(dict(dqc.count_ops()))
        return counter.times(reps // split)

    def controlled_gate_count(self, time: float, reps: int) -> dict[str, int]:
        print(f"KTrotter: Running controlled gate count for time: {time}")

        big_circ = QuantumCircuit(self.ham.num_qubits + 1)
        if reps < SPLIT_SIZE:
            split = 1
        else:
            split = SPLIT_SIZE

        controlled_gate = self.simulation_circuit(time, split).to_gate().control(1)
        big_circ.append(controlled_gate, range(self.ham.num_qubits + 1))
        dqc = self.decomposer.decompose(big_circ)

        counter = Counter()
        counter.add(dict(dqc.count_ops()))
        return counter.times(reps // split)


from ising.hamiltonian.ising_one import parametrized_ising
from ising.hamiltonian.ising_one import trotter_reps
from ising.utils.commutator import commutator_r


def main():
    num_qubits, h = 7, 0.125
    eps = 0.1
    time = 20

    first_ord = trotter_reps(num_qubits, h, time, eps)
    print(f"First Order: {first_ord}")

    for order in range(2, 5, 2):
        hamiltonian = parametrized_ising(num_qubits, h)

        reps = commutator_r(hamiltonian.sparse_repr, order, time, eps)
        print(f"order: {order} reps:{reps}")
        benchmarker = KTrotterBenchmarkTime(hamiltonian, order)

        print(benchmarker.simulation_gate_count(time, reps))
        # print(benchmarker.controlled_gate_count(time, reps))


if __name__ == "__main__":
    main()
