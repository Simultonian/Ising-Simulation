from qiskit import transpile
from qiskit import Aer
from qiskit.providers.aer import AerSimulator
from qiskit.transpiler.passes.synthesis import SolovayKitaev
from qiskit.synthesis import generate_basic_approximations
from qiskit.providers.fake_provider import FakeBackendV2

SINGLE_QUBIT_GATES = ["h", "t", "tdg"]
# ALL_GATES = ["h", "t", "tdg", "s", "sdg", "cx", "rz"]
ALL_GATES = ["cx", "u3"]


class Decomposer:
    def __init__(self, single_qubit_gates=None, all_gates=None):
        if single_qubit_gates is None:
            self.single_qubit_gates = SINGLE_QUBIT_GATES

        if all_gates is None:
            self.all_gates = ALL_GATES

        # self.backend = AerSimulator()

        self.backend = FakeBackendV2()
        # self.backend = Aer.get_backend('aer_simulator')

        approx = generate_basic_approximations(self.single_qubit_gates, depth=3)
        self.skd = SolovayKitaev(recursion_degree=3, basic_approximations=approx)

    def decompose(self, circuit):
        tqc = transpile(circuit, None, optimization_level=1, basis_gates=ALL_GATES)
        return tqc
        return self.skd(tqc)
