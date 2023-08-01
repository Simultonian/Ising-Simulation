from .ising_one import parametrized_ising, trotter_reps


TROTTER_REP_FUNC = {"lie": trotter_reps, "qdrift": trotter_reps}

__all__ = ["parametrized_ising", "trotter_reps", "TROTTER_REP_FUNC"]
