from typing import Union, Optional
import numpy as np
from numpy.typing import NDArray

from qiskit.quantum_info import SparsePauliOp, Operator
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter, EvolutionSynthesis


class QiskitTrotter:
    def __init__(
        self,
        ham: SparsePauliOp,
        observable: Union[SparsePauliOp, NDArray],
        reps: int = 1,
        synthesis: Optional[EvolutionSynthesis] = None,
    ):
        self.num_qubits = ham.num_qubits
        assert isinstance(self.num_qubits, int)

        if isinstance(observable, SparsePauliOp):
            self.observable = observable.to_matrix()
            assert observable is not None
        else:
            self.observable = observable

        if synthesis is None:
            synthesis = LieTrotter(reps=reps)

        self.para_t = Parameter("t")
        evo_gate1 = PauliEvolutionGate(ham, self.para_t, synthesis=synthesis)
        self.circ = QuantumCircuit(self.num_qubits)
        self.circ.append(evo_gate1, range(self.num_qubits))

    def get_unitary(self, t: float) -> NDArray:
        mat = Operator.from_circuit(self.circ.assign_parameters({self.para_t: t})).data
        assert isinstance(mat, np.ndarray)

        return mat

    def get_observations(
        self,
        rho_init: NDArray,
        total_time: float = 0.0,
        division_count: int = 0,
        para_times: list[int] = [],
    ) -> list[float]:
        if division_count == 0:
            if len(para_times) == 0:
                raise ValueError("Both division_count and para_times not provided.")
            times = np.array(para_times)
        else:
            if len(para_times) != 0:
                raise ValueError("Both division_count and para_times provided.")
            times = np.linspace(0, total_time, division_count)

        results = []
        for time in times:
            unitary = self.get_unitary(time)
            rho_final = unitary @ rho_init @ unitary.conj().T
            result = np.trace(np.abs(self.observable @ rho_final))
            results.append(result)

        return results
