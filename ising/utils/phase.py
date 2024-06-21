import numpy as np
from numpy.typing import NDArray
import cmath


def global_phase(a: NDArray):
    # get the phase of the first non-zero value in the matrix
    for row in a:
        for x in row:
            if np.abs(x) > 1e-3:
                theta = cmath.phase(x)
                # phase is e^{i\theta}
                phase = np.exp(0 + 1j * theta)
                # print(x, theta, phase)
                return phase
