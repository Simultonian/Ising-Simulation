import numpy as np
from ising.lindbladian.interaction import RepetitionMap, set_ham_into_pos
from ising.hamiltonian import parametrized_ising
from ising.utils import close_state, global_phase

ENV_SIZE = 1
SYS_SIZE = 2
TOTAL_SIZE = ENV_SIZE + SYS_SIZE
H_VAL = 0.1
TIME = 1
OVERLAP = 0.9


def matrix_exp(eig_vec, eig_val, eig_vec_inv, time: float):
    return eig_vec @ np.diag(np.exp(complex(0, -1) * time * eig_val)) @ eig_vec_inv


def with_eye_env(ham_sys, ham_env, rho_sys, rho_env):
    big_sys_ham = set_ham_into_pos(ham_sys, 0, ENV_SIZE)

    if big_sys_ham.shape[0] != (2**TOTAL_SIZE):
        raise ValueError(f"Incorrect sys size: {big_sys_ham.shape}")

    # each environment is an identity operator for now
    big_env_hams = []
    for env_pos in range(ENV_SIZE):
        # one qubit only
        big_env_ham = set_ham_into_pos(
            ham_env, SYS_SIZE + env_pos, (ENV_SIZE - 1) - env_pos
        )

        if big_env_ham.shape[0] != (2**TOTAL_SIZE):
            raise ValueError(f"Incorrect sys size: {big_env_ham.shape}")
        big_env_hams.append(big_env_ham)

    mapper = RepetitionMap(big_sys_ham, big_env_hams, big_env_hams, 0)
    rho_f = mapper.apply_ri(rho_sys, rho_env, TIME, index=0)
    return rho_f


def without_env(ham, rho_sys):
    rho_f = (
        matrix_exp(ham.eig_vec, ham.eig_val, ham.eig_vec_inv, TIME)
        @ rho_sys
        @ matrix_exp(ham.eig_vec, ham.eig_val, ham.eig_vec_inv, TIME)
    )
    return rho_f


def test_main():
    ham_sys = parametrized_ising(SYS_SIZE, H_VAL)
    env_sys = parametrized_ising(ENV_SIZE, H_VAL)

    init_state = close_state(ham_sys.ground_state, OVERLAP)
    rho_sys = np.outer(init_state, init_state.conj())

    init_state = close_state(env_sys.ground_state, OVERLAP)
    rho_env = np.outer(init_state, init_state.conj())

    eye_env = with_eye_env(ham_sys.matrix, env_sys.matrix, rho_sys, rho_env)
    eye_env = eye_env / global_phase(eye_env)

    no_env = without_env(ham_sys, rho_sys)
    no_env = eye_env / global_phase(no_env)

    np.testing.assert_almost_equal(eye_env, no_env, decimal=2)
    print(f"Match complete")


if __name__ == "__main__":
    test_main()
