from typing import Optional
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp, Pauli


def normalized_eigen(matrix: NDArray, normalize: bool) -> tuple[NDArray, NDArray]:
    eig_val, eig_vec = np.linalg.eig(matrix)
    if normalize:
        lam_0 = np.min(eig_val)
        eig_val -= lam_0

    return (eig_vec, eig_val)


@dataclass
class Hamiltonian:
    sparse_repr: SparsePauliOp
    normalized: bool = False
    _eig_vec: Optional[NDArray] = None
    _eig_val: Optional[NDArray] = None
    _eig_vec_inv: Optional[NDArray] = None
    _matrix: Optional[NDArray] = None
    _ground_state: Optional[NDArray] = None
    _map: Optional[dict[Pauli, complex]] = None
    _spectral_gap: Optional[float] = None

    def __getitem__(self, pauli: Pauli) -> complex:
        if self._map is None:
            self._map = {Pauli(p): v for (p, v) in self.sparse_repr.to_list()}
        return self._map[pauli]

    @property
    def eig_vec(self) -> NDArray:
        if self._eig_vec is None:
            self._matrix = self.sparse_repr.to_matrix()
            assert self._matrix is not None
            self._eig_vec, self._eig_val = normalized_eigen(
                self._matrix, self.normalized
            )
        return self._eig_vec

    @property
    def eig_val(self) -> NDArray:
        if self._eig_val is None:
            self._eig_vec, self._eig_val = normalized_eigen(
                self.sparse_repr.to_matrix(), self.normalized
            )
        return self._eig_val

    @property
    def ground_state(self) -> NDArray:
        if self._ground_state is None:
            ground_pos = np.argmin(self.eig_val)
            # Ground state is the eigenvector corresponding to the smallest eigenvalue
            self._ground_state = self.eig_vec[:, ground_pos]

        return self._ground_state

    @property
    def matrix(self) -> NDArray:
        if self._matrix is None:
            _ = self.eig_vec
        assert self._matrix is not None

        return self._matrix

    @property
    def eig_vec_inv(self) -> NDArray:
        if self._eig_vec_inv is None:
            self._eig_vec_inv = np.linalg.inv(self.eig_vec)
        return self._eig_vec_inv

    @property
    def spectral_gap(self) -> float:
        if self._spectral_gap is None:
            sorted_eigval = sorted(self.eig_val)
            self._spectral_gap = np.abs(sorted_eigval[0] - sorted_eigval[1])
        return self._spectral_gap


def substitute_parameter(ham: Hamiltonian, para: Parameter, val: float) -> Hamiltonian:
    new_sparse_repr = ham.sparse_repr.assign_parameters({para: val})
    return Hamiltonian(sparse_repr=new_sparse_repr, normalized=ham.normalized)
