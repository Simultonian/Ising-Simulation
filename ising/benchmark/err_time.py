import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ising.hamiltonian import Hamiltonian
from ising.hamiltonian import parametrized_ising

from ising.benchmark.sim_function import (
        taylor_gate_count, 
        trotter_gate_count, 
        qdrift_gate_count
        )


def plot_time_error(
    qubit, h_val, start_err_exp, end_err_exp, point_count, obs_norm, time
):
    # TODO
    pass
