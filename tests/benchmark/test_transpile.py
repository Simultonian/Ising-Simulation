from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp

from ising.utils import decompose


def test_decompose():
    X = SparsePauliOp("X")
    Z = SparsePauliOp("Z")
    I = SparsePauliOp("I")

    # build the evolution gate
    operator = (Z ^ Z) - 0.1 * (X ^ X)
    evo = PauliEvolutionGate(operator, time=0.2)

    # plug it into a circuit
    circuit = QuantumCircuit(2)
    circuit.append(evo, range(2))

    dqc = decompose(circuit)
