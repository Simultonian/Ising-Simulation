from qiskit import transpile
from qiskit.providers.aer import AerSimulator
from qiskit.transpiler.passes.synthesis import SolovayKitaev
from qiskit.synthesis import generate_basic_approximations

SINGLE_QUBIT_GATES = ["h", "t", "tdg", "s", "sdg"]
ALL_GATES = ["h", "t", "tdg", "s", "sdg", "cx", "rz"]


def decompose(circuit):
    backend = AerSimulator()
    tqc = transpile(circuit, backend, optimization_level=3, basis_gates=ALL_GATES)

    approx = generate_basic_approximations(SINGLE_QUBIT_GATES, depth=4)
    skd = SolovayKitaev(recursion_degree=2, basic_approximations=approx)

    return skd(tqc)
