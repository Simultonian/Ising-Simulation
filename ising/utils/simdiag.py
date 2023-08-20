import numpy as np


def _degen(tol, vecs, ops, i=0):
    """
    Private function that finds eigen vals and vecs for degenerate matrices..
    """
    if len(ops) == i:
        return vecs

    # New eigenvectors are sometime not orthogonal.
    for j in range(1, vecs.shape[1]):
        for k in range(j):
            dot = vecs[:, j].dot(vecs[:, k].conj())
            if np.abs(dot) > tol:
                vecs[:, j] = (vecs[:, j] - dot * vecs[:, k]) / (
                    1 - np.abs(dot) ** 2
                ) ** 0.5

    subspace = vecs.conj().T @ ops[i] @ vecs
    eigvals, eigvecs = np.linalg.eig(subspace)

    perm = np.argsort(eigvals)
    eigvals = eigvals[perm]

    vecs_new = vecs @ eigvecs[:, perm]
    for k in range(len(eigvals)):
        vecs_new[:, k] = vecs_new[:, k] / np.linalg.norm(vecs_new[:, k])

    k = 0
    while k < len(eigvals):
        ttol = max(tol, tol * abs(eigvals[k]))
        (inds,) = np.where(abs(eigvals - eigvals[k]) < ttol)
        if len(inds) > 1:  # if at least 2 eigvals are degenerate
            vecs_new[:, inds] = _degen(tol, vecs_new[:, inds], ops, i + 1)
        k = inds[-1] + 1
    return vecs_new


def simdiag(ops: list[np.ndarray], *, tol: float = 1e-14):
    """Simultaneous diagonalization of commuting Hermitian matrices.

    Parameters
    ----------
    ops : list of np.NDArrays representing commuting Hermitian operators.

    tol : Tolerance for detecting degenerate eigenstates.

    Returns
    -------
    eigs : Tuple of arrays representing eigvecs and eigvals corresponding to
        simultaneous eigenvectors and eigenvalues for each operator.
    """
    N = ops[0].shape[0]

    eigvals, eigvecs = np.linalg.eig(ops[0])

    k = 0
    while k < N:
        # find degenerate eigenvalues, get indicies of degenerate eigvals
        ttol = max(tol, tol * abs(eigvals[k]))
        (inds,) = np.where(abs(eigvals - eigvals[k]) < ttol)
        if len(inds) > 1:  # if at least 2 eigvals are degenerate
            eigvecs[:, inds] = _degen(tol, eigvecs[:, inds], ops, 1)
        k = inds[-1] + 1

    for k in range(N):
        eigvecs[:, k] = eigvecs[:, k] / np.linalg.norm(eigvecs[:, k])

    eigvals_out = np.zeros((len(ops), N), dtype=np.float64)

    for ind_op, _ in enumerate(ops):
        for j in range(N):
            eigvals_out[ind_op, j] = (
                eigvecs[:, j].T @ ops[ind_op] @ eigvecs[:, j]
            ).real

    return eigvals_out, eigvecs
