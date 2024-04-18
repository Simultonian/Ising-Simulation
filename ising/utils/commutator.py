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

    print(t1, t2, pauli, coeff)
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

    

def alpha_commutator(ham: SparsePauliOp, order: int) -> int:
    """
    Calculates the commutator bound for kth order Trotter using the bounds 
    defined in "Theory of Trotter Error".
    """
    if order != 1 and order % 2 == 1:
        raise ValueError("Not well defined for odd orders")

    # Temporary
    if order > 2:
        raise ValueError("Not defined for higher orders")

    paulis, coeffs = ham.paulis, ham.coeffs
    inds = np.array(list(range(len(paulis))))

    ind_prods = product(inds, repeat=order+1)

    alpha_comm = 0.0
    for cur_term_ind in ind_prods:
        terms = [(paulis[ind], coeffs[ind]) for ind in cur_term_ind]
        val = commute(terms)
        print(terms, val)
        print("-------------")
        alpha_comm += val

    return np.ceil(alpha_comm)


def main():
    x, y = Pauli("X"), Pauli("Y")
    ham = SparsePauliOp([x, y], [1.0, 2.0])
    terms = [(x, 1), (y, 1), (x, 1)]
    res = _commute(terms)
    print(res)
    print(commute(terms))
    alpha_comm = alpha_commutator(ham, 1)
    print(alpha_comm)



if __name__ == "__main__":
    main()
