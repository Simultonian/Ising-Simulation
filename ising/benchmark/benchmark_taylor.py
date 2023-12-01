import json
import numpy as np

from qiskit.quantum_info import Operator
from qiskit.circuit import Parameter

from ising.hamiltonian import parametrized_ising
from ising.observables import overall_magnetization
from ising.utils import close_state
from ising.simulation.taylor import TaylorSingle
from ising.simulation.exact import ExactSimulation


def get_exact(qubit, time, h_val, rho_init, observable):
    h_para = Parameter("h")
    ham = parametrized_ising(qubit, h_para)
    circuit_manager = ExactSimulation(ham, h_para)
    circuit_manager.subsitute_h(h_val)
    circuit_manager.construct_parametrized_circuit()

    return circuit_manager.get_observations(rho_init, observable.matrix, [time])


def get_taylor_results(qubit, time, h_val, rho_init, observable, r_range):
    h_para = Parameter("h")
    ham = parametrized_ising(qubit, h_para)
    # This 0.1 will not be used to calculate `r`.
    circuit_manager = TaylorSingle(ham, h_para, 0.1)
    circuit_manager.subsitute_h(h_val)
    circuit_manager.construct_parametrized_circuit()

    results = []
    for r in r_range:
        ans = circuit_manager.get_observation(rho_init, observable.matrix, time, r)[0]
        results.append(ans)

    return results


def main():
    qubit = 8
    time = 10
    start_r, end_r = 1, 4
    r_count = 10
    h_val = 0.1
    overlap = 0.8

    r_range = [int(10**x) for x in np.linspace(start_r, end_r, r_count)]

    observable = overall_magnetization(qubit)

    h_para = Parameter("h")
    ham = parametrized_ising(qubit, h_para)
    exact = ExactSimulation(ham, h_para)
    exact.subsitute_h(h_val)
    exact.construct_parametrized_circuit()

    ground_state = exact.ham_subbed.ground_state
    init_state = close_state(ground_state, overlap)
    rho_init = np.outer(init_state, init_state.conj().T)
    rho_init = np.array(Operator(rho_init).reverse_qargs().data)

    exact_ans = exact.get_observations(rho_init, observable.matrix, [time])[0]
    results = get_trotter_results(qubit, time, h_val, rho_init, observable, r_range)

    rel_diff = [abs(result - exact_ans) / exact_ans for result in results]

    file_name = f"data/simulation/benchmark_magnetization_lie_{qubit}.json"
    save_dict = {
        "qubit": qubit,
        "time": time,
        "h_val": h_val,
        "r_range": r_range,
        "results": results,
        "exact_ans": exact_ans,
        "rel_diff": rel_diff,
    }

    print(save_dict)
    print(f"Saving results at: {file_name}")
    with open(file_name, "w") as file:
        json.dump(save_dict, file)


def test_main():
    main()


if __name__ == "__main__":
    np.random.seed(42)
    main()
