from typing import Union
from functools import cache
import numpy as np
from tqdm import tqdm

from ising.hamiltonian import Hamiltonian
from ising.groundstate.simulation.utils import (
    ground_state_constants,
    ground_state_maximum_time,
)
from ising.utils import Decomposer
from ising.utils.gate_count import count_non_trivial
from ising.benchmark.gates.counter import Counter

from qiskit.circuit.library import PauliEvolutionGate, PauliGate
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, SparsePauliOp, PauliList
from qiskit.circuit.library import PauliGate
from ising.simulation.taylor.utils import (
    get_alphas,
)


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
        k = np.floor(
            np.log(self.beta * time / self.error)
            / np.log(np.log(self.beta * time / self.error))
        )
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


def theta_m(t_bar, r, k):
    val = np.arccos(((1 + (t_bar / r) ** 2) / (k + 1)) ** (-1 / 2))
    return val


SPLIT_SIZE = 100


class TaylorBenchmarkTime:
    def __init__(self, ham: Hamiltonian):
        """
        Benchmark calculator for Trotterization based groundstate preparation.

        Inputs:
            - ham: Hamitlonian
        """

        self.ham = ham
        self.num_qubits = ham.num_qubits
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
        self.coeffs = self.coeffs / self.lambd
        self.paulis = paulis
        self.sign_map = sign_map
        self.indices = list(range(len(self.paulis)))

    def simulation_circuit(
        self, t_bar: float, reps: int, k_max: int, split: int
    ) -> QuantumCircuit:
        """
        Calculates the gate depth for given time
        """
        circuit = QuantumCircuit(self.ham.num_qubits)

        alphas = get_alphas(t_bar, k_max, reps)
        k_probs = np.abs(alphas)
        k_probs /= np.sum(k_probs)

        # Could be heavy operation for large reps.
        print(f"reps:{reps}, k_max:{k_max}")
        for _ in range(split):
            k = np.random.choice(k_max + 1, p=k_probs)

            samples = np.random.choice(self.indices, p=self.coeffs, size=k + 1)

            for sample in samples[:-1]:
                pauli = self.paulis[sample]
                circuit.append(PauliGate(pauli.to_label()), range(self.num_qubits))

            evo = PauliEvolutionGate(
                self.paulis[samples[-1]], time=theta_m(t_bar, reps, k)
            )
            circuit.append(evo, range(evo.num_qubits))

        return circuit

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

    def simulation_gate_count(self, time: float, k: int) -> dict[str, int]:
        print(f"Taylor: Running gate count for time: {time}")
        t_bar = time * self.lambd
        reps = max(20, int(10 * np.ceil(t_bar) ** 2))

        if reps < SPLIT_SIZE:
            split = 1
        else:
            split = SPLIT_SIZE

        circuit = self.simulation_circuit(t_bar, reps, k, split)
        dqc = self.decomposer.decompose(circuit)

        counter = Counter()
        counter.add(dict(dqc.count_ops()))
        return counter.times(reps // split)

    def controlled_gate_count(self, time: float, k: int) -> dict[str, int]:
        print(f"Taylor: Running controlled gate count for time: {time}")
        t_bar = time * self.lambd
        reps = max(20, int(10 * np.ceil(t_bar) ** 2))

        if reps < SPLIT_SIZE:
            split = 1
        else:
            split = SPLIT_SIZE

        big_circ = QuantumCircuit(self.ham.num_qubits + 1)
        controlled_gate = (
            self.simulation_circuit(time, split, k, split).to_gate().control(1)
        )
        big_circ.append(controlled_gate, range(self.ham.num_qubits + 1))
        dqc = self.decomposer.decompose(big_circ)

        counter = Counter()
        counter.add(dict(dqc.count_ops()))
        return counter.times(reps // split)


from ising.hamiltonian.ising_one import parametrized_ising


def main():
    num_qubits, h = 10, 0.125
    eps = 0.1
    time = 10
    hamiltonian = parametrized_ising(num_qubits, h)
    lambd = np.sum(np.abs(hamiltonian.coeffs))
    k = int(np.floor(np.log(lambd * time / eps) / np.log(np.log(lambd * time / eps))))

    benchmarker = TaylorBenchmarkTime(hamiltonian)

    print(benchmarker.simulation_gate_count(time, k))
    print(benchmarker.controlled_gate_count(time, k))


if __name__ == "__main__":
    main()
