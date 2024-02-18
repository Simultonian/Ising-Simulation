from ising.benchmark.gates.trotter import TrotterBenchmark
from ising.hamiltonian import parametrized_ising
from qiskit.quantum_info import Pauli
from ising.utils.decompose import Decomposer
from qiskit.circuit.library import PauliEvolutionGate
from qiskit import QuantumCircuit


def gates_for_pauli(pauli, time) -> dict[str, int]:
    evo = PauliEvolutionGate(pauli, time=time)
    circuit = QuantumCircuit(evo.num_qubits)
    circuit.append(evo, range(evo.num_qubits))
    decomposer = Decomposer()
    dqc = decomposer.decompose(circuit)
    return dict(dqc.count_ops())


def test_gate_count_pauli():
    count = gates_for_pauli(Pauli("ZZ"), 20)


def test_ising_trotter_count():
    ham = parametrized_ising(4, 0.1)
    obs_norm, overlap, error, success = 1, 0.7, 0.1, 0.1

    benchmarker = TrotterBenchmark(
        ham, obs_norm, overlap=overlap, error=error, success=success
    )
    gates = benchmarker.calculate_gates()
