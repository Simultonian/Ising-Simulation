from qiskit import transpile
from qiskit.providers.aer import AerSimulator
from qiskit.transpiler.passes.synthesis import SolovayKitaev
from qiskit.synthesis import generate_basic_approximations

SINGLE_QUBIT_GATES = ["h", "t", "tdg"]
ALL_GATES = ["h", "t", "tdg", "s", "sdg", "cx", "rz"]


class Decomposer:
    def __init__(self, single_qubit_gates=None, all_gates=None):
        if single_qubit_gates is None:
            self.single_qubit_gates = SINGLE_QUBIT_GATES

        if all_gates is None:
            self.all_gates = ALL_GATES

        self.backend = AerSimulator()

        approx = generate_basic_approximations(self.single_qubit_gates, depth=3)
        self.skd = SolovayKitaev(recursion_degree=3, basic_approximations=approx)

    def decompose(self, circuit):
        tqc = transpile(
            circuit, self.backend, optimization_level=3, basis_gates=ALL_GATES
        )
        return self.skd(tqc)
