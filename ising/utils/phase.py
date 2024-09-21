import numpy as np
from numpy.typing import NDArray
import cmath


def global_phase(a: NDArray):
    # get the phase of the first non-zero value in the matrix
    phase = 1
    for row in a:
        for x in row:
            if np.abs(x) > 1e-3:
                theta = cmath.phase(x)
                if abs(theta) < 1e-6:
                    theta = 0
                # phase is e^{i\theta}
                return np.exp(0 + 1j * theta)

    return phase
