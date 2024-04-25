import numpy as np
from itertools import product
from collections import defaultdict
from qiskit.quantum_info import SparsePauliOp, Pauli


def balance_prod(t1: tuple[Pauli, float], t2: tuple[Pauli, float]):
    pa, ca = t1
    pb, cb = t2

    coeff = ca * cb
    pauli = pa @ pb

    if pauli.to_label()[0] == "-":
        coeff *= -1
        pauli = pauli * -1

    return (pauli, coeff)


def _commute(terms: list[tuple[Pauli, float]]) -> dict[Pauli, float]:
    if len(terms) == 2:
        a, b = terms
        tail = defaultdict(float)

        p, c = balance_prod(a, b)
        tail[p] += c

        p, c = balance_prod(b, a)
        tail[p] -= c
        return tail

    first, rem = terms[0], terms[1:]
    tail = _commute(rem)
    new_tail = defaultdict(float)

    for second in tail.items():
        p, c = balance_prod(first, second)
        new_tail[p] += c

        p, c = balance_prod(second, first)
        new_tail[p] -= c

    return new_tail


def commute(terms: list[tuple[Pauli, float]]) -> float:
    """
    Calculates [H_1, ... [H_k-1, H_k]] and returns the norm
    """
    res = _commute(terms)
    coeffs = np.array(list(res.values()))
    return np.sum(np.abs(coeffs))


from tqdm import tqdm


def alpha_commutator(ham: SparsePauliOp, order: int) -> int:
    """
    Calculates the commutator bound for kth order Trotter using the bounds
    defined in "Theory of Trotter Error".
    """
    if order != 1 and order % 2 == 1:
        raise ValueError("Not well defined for odd orders")

    paulis, coeffs = ham.paulis, ham.coeffs
    inds = np.array(list(range(len(paulis))))

    ind_prods = product(inds, repeat=order + 1)

    total_count = len(inds) ** (order + 1)
    alpha_comm = 0.0

    # with tqdm(total=total_count) as pbar:
    for cur_term_ind in ind_prods:
        terms = [(paulis[ind], coeffs[ind]) for ind in cur_term_ind]
        val = commute(terms)
        alpha_comm += val
            # pbar.update(1)

    return np.ceil(alpha_comm)


def alpha_commutator_second_order(ham: SparsePauliOp) -> int:
    """
    We use three explicit loops for calculating the commutator bound, to avoid
    miscalculations. Any higher trotter bound can not be calculated.
    """
    paulis, coeffs = ham.paulis, ham.coeffs
    inds = np.array(list(range(len(paulis))))

    total_count = len(inds) ** 3

    alpha_comm = 0.0
    # with tqdm(total=total_count) as pbar:
    for ia in inds:
        for ib in inds:
            for ic in inds:
                tail = defaultdict(float)
                # abc - acb - bca + cba
                # pbar.update(1)

                a, b, c = (
                    (paulis[ia], coeffs[ia]),
                    (paulis[ib], coeffs[ib]),
                    (paulis[ic], coeffs[ic]),
                )

                # abc
                p, ce = balance_prod(a, balance_prod(b, c))
                tail[p] += ce

                # -acb
                p, ce = balance_prod(a, balance_prod(c, b))
                tail[p] -= ce

                # -bca
                p, ce = balance_prod(b, balance_prod(c, a))
                tail[p] -= ce

                # cba
                p, ce = balance_prod(c, balance_prod(b, a))
                tail[p] += ce
                cur_coeffs = np.array(list(tail.values()))
                alpha_comm += np.sum(np.abs(cur_coeffs))

    return alpha_comm


def alpha_commutator_first_order(ham: SparsePauliOp) -> int:
    """
    We use three explicit loops for calculating the commutator bound, to avoid
    miscalculations. Any higher trotter bound can not be calculated.
    """
    paulis, coeffs = ham.paulis, ham.coeffs
    inds = np.array(list(range(len(paulis))))

    total_count = len(inds) ** 2

    alpha_comm = 0.0
    # with tqdm(total=total_count) as pbar:
    for ia in inds:
        for ib in inds:
            tail = defaultdict(float)
            # ab - ba
            # pbar.update(1)

            a, b = (paulis[ia], coeffs[ia]), (paulis[ib], coeffs[ib])

            # ab
            p, ce = balance_prod(a, b)
            tail[p] += ce

            # -ba
            p, ce = balance_prod(b, a)
            tail[p] -= ce

            cur_coeffs = np.array(list(tail.values()))
            alpha_comm += np.sum(np.abs(cur_coeffs))

    return alpha_comm


def commutator_r(ham: SparsePauliOp, order: int, time: float, error: float) -> int:
    """
    Calculates a tighter bound for higher order trotter using the alpha
    commutator from "Theory of Trotter Error" results.
    """
    alpha_com = alpha_commutator(ham, order)
    nr = (alpha_com ** (1 / order)) * (time ** (1 + 1 / order))
    dr = error ** (1 / order)
    return np.ceil(nr / dr)


def commutator_r_first_order(
    ham: SparsePauliOp, time: float, error: float, alpha_com: float = -1
) -> int:
    """
    Same as `commutator_r` but it is hard-coded with two for loops
    """
    if alpha_com == -1:
        alpha_com = alpha_commutator_first_order(ham)
    nr = alpha_com * (time**2)
    dr = error
    return np.ceil(nr / dr)


def commutator_r_second_order(
    ham: SparsePauliOp, time: float, error: float, alpha_com: float = -1
) -> int:
    """
    Same as `commutator_r` but it is hard-coded with three for loops
    """
    if alpha_com == -1:
        alpha_com = alpha_commutator_second_order(ham)
    nr = (alpha_com ** (1 / 2)) * (time ** (1 + 1 / 2))
    dr = error ** (1 / 2)
    return np.ceil(nr / dr)


from ising.hamiltonian.ising_one import parametrized_ising
from ising.hamiltonian.ising_one import trotter_reps, trotter_reps_general
from ising.hamiltonian import parse


def main():
    num_qubits, h = 7, 0.125
    eps = 0.1
    time = 20

    name = "methane"
    hamiltonian = parse(name)

    norm_first_ord = trotter_reps(num_qubits, h, time, eps)
    print(f"First Order Non-Commutator: {norm_first_ord}")

    norm_first_ord = trotter_reps_general(hamiltonian.sparse_repr, time, eps)
    print(f"First Order General: {norm_first_ord}")

    first_ord = commutator_r_first_order(hamiltonian.sparse_repr, time, eps)
    print(f"First Order: {first_ord}")

    second_ord = commutator_r_second_order(hamiltonian.sparse_repr, time, eps)
    print(f"Second Order: {second_ord}")


if __name__ == "__main__":
    main()
