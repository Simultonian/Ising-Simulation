from qiskit.quantum_info import Pauli


def count_non_trivial(pauli: Pauli) -> int:
    """
    Counts the number of non-Identity operators in the multi-qubit Pauli
    operator
    """
    count = 0
    for p in pauli:
        if str(p) == "I":
            continue
        count += 1

    return count


def cx_count(pauli: Pauli) -> int:
    """
    Counts the number of non-Identity operators in the multi-qubit Pauli
    operator and provide an upper bound for cx gate count
    """
    count = 0
    for p in pauli:
        if str(p) == "I":
            continue
        count += 1

    return 2 * (count - 1)
