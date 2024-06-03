import numpy as np
from ising.hamiltonian import Hamiltonian, parametrized_ising
from qiskit.quantum_info import Pauli, SparsePauliOp


def transpose(sparse: SparsePauliOp) -> SparsePauliOp:
    """
    Take the Hamiltonian and apply transpose operation on it.

    #NOTE: For real coefficients, it is equivalent to complex conjugate
    operation as well.
    """

    new_coeffs = []

    for pauli, coeff in zip(sparse.paulis, sparse.coeffs):
        for p in str(pauli):
            if p == "Y":
                coeff *= -1
        new_coeffs.append(coeff)

    return SparsePauliOp(data=sparse.paulis, coeffs=new_coeffs)


LOWERING = np.array([[0, 0], [1, 0]])


def lowering_all_sites(chain_size: int, gamma: float = 1):
    """
    Returns the list for lindbladian operators for the lowering operator
    on all sites.
    """

    l_ops = []
    for site in range(chain_size):
        l, r = site, chain_size - (site + 1)

        if l == 0:
            op = LOWERING.copy()
        else:
            op = np.kron(np.eye(2**l), LOWERING)

        if r > 0:
            op = np.kron(op, np.eye(2**r))

        l_ops.append(np.sqrt(gamma) * op)

    return l_ops


def lindbladian_operator(hamiltonian, lindbladian_ops):
    """
    Gives back the Lindbladian operator for the given system Hamiltonian and
    the Lindbladian operators

    Inputs:
        - hamiltonian: Matrix of system Hamiltonian
        - lindbladian_ops: Lindbladian operators
    """

    identity = np.eye(hamiltonian.shape[0])

    term1 = -1j * np.kron(identity, hamiltonian)

    # Transpose of Ising Chain does not make any change
    term2 = 1j * hamiltonian.kron(identity)

    term3 = np.zeros_like(term1)

    for l_op in lindbladian_ops:
        t1 = l_op.conj().kron(l_op)
        t2 = np.kron(identity, (l_op.T.conj() @ l_op))
        t3 = (l_op.T @ l_op.conj()).kron(identity)

        term3 += t1 - 0.5 * (t2 + t3)

    return term1 + term2 + term3
