from ising.hamiltonian import Hamiltonian
from qiskit.quantum_info import SparsePauliOp
import json


def parse(name) -> Hamiltonian:
    """Parses the molecule present in 'data/molecules/'

    Input: Name of the molecule, the file name will be {name}.json

    Returns: Hamiltonian
    """

    data = {}
    with open(f"data/molecules/{name}.json", "r") as json_file:
        data = json.load(json_file)

    # eV to Hartree
    gap = abs(float(data["homo"]) - float(data["lumo"])) * 0.0367493

    coeffs = []
    for rl, im in zip(data["real"], data["imag"]):
        coeffs.append(rl + 1j * im)

    sparse_repr = SparsePauliOp(data=data["terms"], coeffs=coeffs)
    return Hamiltonian(
        sparse_repr=sparse_repr, _spectral_gap=gap, _approx_spectral_gap=gap
    )
