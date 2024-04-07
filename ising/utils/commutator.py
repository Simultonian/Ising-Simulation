import numpy as np
from itertools import product
from collections import defaultdict
from qiskit.quantum_info import SparsePauliOp, Pauli

def _commute(terms) -> dict[Pauli, float]:
    if len(terms) == 2:
        a, b = terms
        pa, ca = a
        pb, cb = b
        tail = defaultdict(float)
        if ca * cb < 0:
            tail[-pa @ pb] += -ca * cb
            tail[pb @ pa] += -ca * cb
        else:
            tail[pa @ pb] += ca * cb
            tail[-pb @ pa] += ca * cb

        return tail

    first, rem = terms[0], terms[1:]
    tail = _commute(rem)
    new_tail = defaultdict(float)

    pa, ca = first
    for pb, cb in tail.items():
        if ca * cb > 0:
            new_tail[pa @ pb] += ca * cb
            new_tail[-pb @ pa] += ca * cb
        else:
            new_tail[-pa @ pb] += -ca * cb
            new_tail[pb @ pa] += -ca * cb

    return new_tail


def commute(terms: list[tuple[Pauli, float]]) -> float:
    """
    Calculates [H_1, ... [H_k-1, H_k]] and returns the norm
    """
    res = _commute(terms)
    paulis = list(res.keys())
    coeffs = [res[pauli] for pauli in paulis]

    ham = SparsePauliOp(paulis, coeffs)

    

def alpha_commutator(ham: SparsePauliOp, order: int, time:float, error: float) -> int:
    """
    Calculates the commutator bound for kth order Trotter using the bounds 
    defined in "Theory of Trotter Error".
    """
    if order != 1 and order % 2 == 1:
        raise ValueError("Not well defined for odd orders")

    # Temporary
    if order > 2:
        raise ValueError("Not defined for ")

    paulis, coeffs = ham.paulis, ham.coeffs
    inds = np.array(list(range(len(paulis))))

    ind_prods = product(inds, repeat=order+1)

    # for ind in ind_prods:
        # 

def main():
    x, y = Pauli("X"), Pauli("Y")
    terms = [(x, 1), (y, 1), (x, 1)]
    res = _commute(terms)
    paulis = list(res.keys())
    coeffs = [res[pauli] for pauli in paulis]

    print(paulis, coeffs)
    ham = SparsePauliOp(paulis, coeffs)
    print(ham)



if __name__ == "__main__":
    main()
