from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Pauli
from qiskit import transpile

from ising.utils.gate_count import count_non_trivial

ALL_GATES = ["cx", "u3"]

def test_cs_count_zz():
    zz = Pauli("ZZ")
    time = 5.0
    evo = PauliEvolutionGate(zz, time)

    circuit = QuantumCircuit(2)
    circuit.append(evo, range(2))

    tqc = transpile(circuit, None, optimization_level=3, basis_gates=ALL_GATES)

    counter = tqc.count_ops()
    assert counter["cx"] == 2

def test_cs_count_xx():
    xx = Pauli("XX")
    time = 100.0
    evo = PauliEvolutionGate(xx, time)

    circuit = QuantumCircuit(2)
    circuit.append(evo, range(2))

    tqc = transpile(circuit, None, optimization_level=3, basis_gates=ALL_GATES)

    counter = tqc.count_ops()
    assert counter["cx"] == 2

def test_cs_count_xxixxix():
    pauli = Pauli("XXIXXIX")
    time = 100.0
    evo = PauliEvolutionGate(pauli, time)

    circuit = QuantumCircuit(pauli.num_qubits)
    circuit.append(evo, range(pauli.num_qubits))

    tqc = transpile(circuit, None, optimization_level=3, basis_gates=ALL_GATES)

    counter = tqc.count_ops()

    count = count_non_trivial(pauli)
    actual = 2 * (count - 1)
    assert counter["cx"] == actual

def test_cs_count_double():
    paulis = [Pauli("XXIXXIX"), Pauli("IXIXIII")]
    time = 100.0
    circuit = QuantumCircuit(paulis[0].num_qubits)
    for pauli in paulis:
        evo = PauliEvolutionGate(pauli, time)
        circuit.append(evo, range(pauli.num_qubits))

    # Optimization could cancel out gates, it doesn't in this case
    tqc = transpile(circuit, None, optimization_level=3, basis_gates=ALL_GATES)

    counter = tqc.count_ops()

    total = 0
    for pauli in paulis:
        count = count_non_trivial(pauli)
        total += 2 * (count - 1)

    assert counter["cx"] == total


def test_cx_controlled():
    pauli = Pauli("XX")
    time = 100.0
    controlled_gate = PauliEvolutionGate(pauli, time).control(1)

    circuit = QuantumCircuit(3)
    circuit.append(controlled_gate, range(pauli.num_qubits + 1))

    tqc = transpile(circuit, None, optimization_level=3, basis_gates=ALL_GATES)

    counter = tqc.count_ops()

    count = count_non_trivial(pauli)
    actual = 2 * (count - 1)
    assert counter["cx"] == actual
