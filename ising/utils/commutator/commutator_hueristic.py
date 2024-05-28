import numpy as np
from collections import defaultdict
from tqdm import tqdm

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


def alpha_commutator_second_order(ham: SparsePauliOp, cutoff: float = 0) -> int:
    """
    We use three explicit loops for calculating the commutator bound, to avoid
    miscalculations. Any higher trotter bound can not be calculated.
    """
    paulis, coeffs = ham.paulis, ham.coeffs
    inds = np.array(list(range(len(paulis))))

    total_count = len(inds) ** 3

    sorted_pairs = list(
        sorted(zip(paulis, coeffs), key=lambda x: abs(x[1]), reverse=True)
    )
    alpha_comm = 0.0

    max_term = sorted_pairs[0][1]

    norm = 0.0
    for x, y in sorted_pairs:
        norm += abs(y)

    with tqdm(total=total_count) as pbar:
        for ia in inds:
            ta = sorted_pairs[ia][1]
            if abs(ta * (max_term**2)) < cutoff:
                break

            for ib in inds:
                tb = sorted_pairs[ib][1]
                if abs(ta * tb * max_term) < cutoff:
                    break

                for ic in inds:
                    tail = defaultdict(float)
                    # abc - acb - bca + cba
                    pbar.update(1)

                    a, b, c = sorted_pairs[ia], sorted_pairs[ib], sorted_pairs[ic]

                    # abc
                    p, ce = balance_prod(a, balance_prod(b, c))
                    if abs(ce) < cutoff:
                        break

                    tail[p] += ce

                    # -acb
                    p, ce = balance_prod(a, balance_prod(c, b))
                    tail[p] -= ce
                    if abs(ce) < cutoff:
                        break

                    # -bca
                    p, ce = balance_prod(b, balance_prod(c, a))
                    tail[p] -= ce
                    if abs(ce) < cutoff:
                        break

                    # cba
                    p, ce = balance_prod(c, balance_prod(b, a))
                    tail[p] += ce
                    if abs(ce) < cutoff:
                        break

                    cur_coeffs = np.array(list(tail.values()))
                    alpha_comm += np.sum(np.abs(cur_coeffs))

    return alpha_comm


def _pauli_commute(a: Pauli, b: Pauli):
    x1, z1 = a._x, a._z

    a_dot_b = np.mod((x1 & b._z).sum(axis=1), 2)
    b_dot_a = np.mod((b._x & z1).sum(axis=1), 2)

    return a_dot_b == b_dot_a


def alpha_commutator_first_order(
    sorted_pairs: list[tuple[Pauli, float]], error: float, delta: float, cutoff_count
) -> float:
    """
    We run two explicit loops and cutoff early if the number of iterations have
    been exhausted. It also terminates early if the delta requirement is met.
    """
    inds = np.array(list(range(len(sorted_pairs))))

    total_count = len(inds) ** 2

    hmax = sorted_pairs[0][1]

    alpha_u = total_count * (hmax**2)
    alpha_l = 0
    cur_count = 0

    with tqdm(total=total_count) as pbar:
        for ia in inds:
            for ib in inds:
                # ab - ba
                pbar.update(1)

                a, b = sorted_pairs[ia], sorted_pairs[ib]
                ce = abs(b[1] * a[1])

                cur_count += 1
                alpha_u -= hmax**2
                if _pauli_commute(a[0], b[0]):
                    alpha_l += ce
                    alpha_u += ce

                print(f"{alpha_l} : {alpha_u}")
                if alpha_l > 0 and (alpha_u / alpha_l) - 1 < (delta / error):
                    return alpha_l
                if cur_count == cutoff_count:
                    return alpha_u

    assert cur_count == total_count
    assert alpha_l == alpha_u

    return alpha_u


def r_first_order(
    sorted_pairs: list[tuple[Pauli, float]], time: float, error: float, **kwargs
) -> int:
    """
    Same as `commutator_r` but it is hard-coded with two for loops
    """
    alpha_com = kwargs.get("alpha_com", -1)
    cutoff_count = kwargs.get("cutoff_count", 0)
    delta = kwargs.get("delta", 0)
    if alpha_com == -1:
        alpha_com = alpha_commutator_first_order(
            sorted_pairs, error, delta, cutoff_count
        )

    nr = alpha_com * (time**2)
    dr = error
    return np.ceil(nr / dr)


def r_second_order(ham: SparsePauliOp, time: float, error: float, **kwargs) -> int:
    """
    Same as `commutator_r` but it is hard-coded with three for loops
    """
    alpha_com = kwargs.get("alpha_com", -1)
    cutoff = kwargs.get("cutoff", 0)

    if alpha_com == -1:
        alpha_com = alpha_commutator_second_order(ham, cutoff)

    nr = (alpha_com ** (1 / 2)) * (time ** (1 + 1 / 2))
    dr = error ** (1 / 2)
    return np.ceil(nr / dr)


def main():
    from ising.hamiltonian.ising_one import parametrized_ising
    from ising.hamiltonian.ising_one import trotter_reps, trotter_reps_general
    from ising.hamiltonian import parse

    num_qubits, h = 7, 0.125
    eps = 0.1
    time = 1.0

    # name = "methane"
    # hamiltonian = parse(name)
    hamiltonian = parametrized_ising(num_qubits, h)

    sorted_pairs = list(
        sorted(
            zip(hamiltonian.paulis, hamiltonian.coeffs),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
    )

    sorted_pairs = [(x, y.real) for (x, y) in sorted_pairs]

    # norm_first_ord = trotter_reps(num_qubits, h, time, eps)
    # print(f"First Order Non-Commutator: {norm_first_ord}")

    # norm_first_ord = trotter_reps_general(hamiltonian.sparse_repr, time, eps)
    # print(f"First Order General: {norm_first_ord}")

    first_ord = r_first_order(sorted_pairs, time, eps, cutoff=time)
    print(f"First Order: {first_ord}")
    return

    second_ord = r_second_order(hamiltonian.sparse_repr, time, eps, cutoff=time)
    print(f"Second Order: {second_ord}")


if __name__ == "__main__":
    main()
