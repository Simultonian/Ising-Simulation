import numpy as np


def close_state(state, overlap, other_states=None):
    """
    Construct a state |y> `a` close to the given state |x>

    |y> = a * |x> + (1 - a) * sum_i 1/S * states_i

    Then normalize the state by calculating the inner product
    """

    if other_states is None:
        plus = np.ones_like(state) / np.sqrt(len(state))
        other_states = np.array([plus])

    state = overlap * state + (1 - overlap) * np.mean(other_states, axis=0)

    norm = np.sqrt(np.sum(np.abs(state) ** 2))
    state = state / norm

    return state
