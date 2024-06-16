import numpy as np
import json
from ising.hamiltonian import parametrized_ising
from ising.utils import close_state
from ising.observables import overall_magnetization
from ising.lindbladian.unraveled import lowering_all_sites, lindbladian_operator

# log scale
GAMMA_RANGE = (0, -4)
GAMMA_COUNT = 4

# not log scale
TIME_RANGE = (1, 5)
TIME_COUNT = 10

# system params
CHAIN_SIZE = 5
H_VAL = -0.1

# simulation params
OVERLAP = 0.8


def _round(mat):
    return np.round(mat, decimals=3)


def matrix_exp_no_i(eig_vec, eig_val, eig_vec_inv, time: float):
    return eig_vec @ np.diag(np.exp(time * eig_val)) @ eig_vec_inv


def matrix_exp(eig_vec, eig_val, eig_vec_inv, time: float):
    return eig_vec @ np.diag(np.exp(complex(0, -1) * time * eig_val)) @ eig_vec_inv


def reshape_vec(vec):
    l = vec.shape[0]
    assert vec.shape[1] == 1
    d = int(np.sqrt(l))

    rho = np.zeros((d, d), dtype=complex)

    for i, x in enumerate(vec):
        r, c = i % d, i // d
        rho[r][c] = x[0]

    return rho


def test_main():
    gamma_points = [
        10**x for x in np.linspace(GAMMA_RANGE[0], GAMMA_RANGE[1], GAMMA_COUNT)
    ]
    gamma_points.append(0)
    time_points = [x for x in np.linspace(TIME_RANGE[0], TIME_RANGE[1], TIME_COUNT)]
    observable = overall_magnetization(CHAIN_SIZE).matrix

    magnetization = {}
    print(f"Running for {GAMMA_COUNT} gamma_points and {TIME_COUNT} time_points")
    ising = parametrized_ising(CHAIN_SIZE, H_VAL)
    ising_matrix = ising.matrix

    ground_state = ising.ground_state
    init_state = close_state(ground_state, OVERLAP)
    rho_init = np.outer(init_state, init_state.conj())
    rho_vec = rho_init.reshape(-1, 1)

    for gamma in gamma_points:
        magnetization[gamma] = {}

        cks = lowering_all_sites(CHAIN_SIZE, gamma=gamma)
        l_op = lindbladian_operator(ising_matrix, cks)

        # eigenvalue manipulation
        eig_val, eig_vec = np.linalg.eig(l_op)
        eig_vec_inv = np.linalg.inv(eig_vec)

        for time in time_points:
            op_time_matrix = matrix_exp_no_i(eig_vec, eig_val, eig_vec_inv, time)
            rho_vec_final = op_time_matrix @ rho_vec
            rho_final = reshape_vec(rho_vec_final)
            result = np.trace(np.abs(observable @ rho_final))

            magnetization[gamma][time] = result

            print(f"gamma:{gamma}, time:{time}, RES: {result}")

        print("------------------------------------------")

    save = {"META": {"H_VAL": H_VAL, "OVERLAP": 0.9}, "ANSWERS": magnetization}
    file_name = f"data/lindbladian/time_vs_magn_gamma/size_{CHAIN_SIZE}.json"

    with open(file_name, "w") as file:
        json.dump(save, file)
    print(f"saved to {file_name}")


if __name__ == "__main__":
    test_main()
