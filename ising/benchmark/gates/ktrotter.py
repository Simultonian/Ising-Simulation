from typing import Union
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
from qiskit.quantum_info import Pauli, SparsePauliOp

from ising.benchmark.gates.counter import Counter
from qiskit.synthesis import SuzukiTrotter

SPLIT_SIZE = 100

class TrotterBenchmarkTime:
    def __init__(self, ham: Hamiltonian):
        """
        Benchmark calculator for Trotterization based groundstate preparation.

        Inputs:
            - ham: Hamitlonian
        """

        self.ham = ham
        self.decomposer = Decomposer()
        self.synth = SuzukiTrotter()


    def simulation_circuit(self, time: float, reps: int) -> QuantumCircuit:
        """
        Calculates the gate depth for given time
        """
        circuit = QuantumCircuit(self.ham.num_qubits)

        evo = PauliEvolutionGate(
                self.ham.sparse_repr, 
                time = time / reps,
                synthesis=self.synth
                )
        circuit.append(evo, range(evo.num_qubits))

        # Could be heavy operation for large reps.
        circuit = circuit.repeat(reps)
        return circuit

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
def main():
    num_qubits, h = 3, 0.125
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
