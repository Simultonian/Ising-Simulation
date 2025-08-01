from typing import Optional
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import linalg
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
    _approx_spectral_gap: Optional[float] = None
    normalized: bool = False
    _eig_vec: Optional[NDArray] = None
    _eig_val: Optional[NDArray] = None
    _eig_vec_inv: Optional[NDArray] = None
    _matrix: Optional[NDArray] = None
    _ground_state: Optional[NDArray] = None
    _map: Optional[dict[Pauli, complex]] = None
    _spectral_gap: Optional[float] = None
    _paulis: tuple[Pauli] = tuple([])
    _coeffs: tuple[float] = tuple([])

    def __getitem__(self, pauli: Pauli) -> complex:
        if self._map is None:
            self._map = {Pauli(p): v for (p, v) in self.sparse_repr.to_list()}
        return self._map[pauli]

    @property
    def approx_spectral_gap(self) -> float:
        if self._approx_spectral_gap is None:
            self._approx_spectral_gap = sparse_spectral_gap(self)
        return self._approx_spectral_gap

    @property
    def num_qubits(self) -> int:
        return self.sparse_repr.num_qubits

    @property
    def map(self) -> dict[Pauli, complex]:
        if self._map is None:
            self._map = {Pauli(p): v for (p, v) in self.sparse_repr.to_list()}
        return self._map

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
            self._matrix = self.eig_vec @ np.diag(self.eig_val) @ self.eig_vec_inv
        assert self._matrix is not None

        return self._matrix

    @property
    def eig_vec_inv(self) -> NDArray:
        if self._eig_vec_inv is None:
            self._eig_vec_inv = np.linalg.inv(self.eig_vec)
        return self._eig_vec_inv

    @property
    def spectral_gap(self) -> float:
        """
        WARNING: This code is flawed since due to precision error and
        degeneracy.
        """
        if self._spectral_gap is None:
            rounded = list(set(np.round(self.eig_val, decimals=8).real.tolist()))
            rounded.sort()
            self._spectral_gap = np.abs(rounded[0] - rounded[1])
        return self._spectral_gap

    @property
    def paulis(self) -> tuple[Pauli]:
        if len(self._paulis) == 0:
            self._paulis = tuple([Pauli(p) for (p, _) in self.sparse_repr.to_list()])
            self._coeffs = tuple([v for (_, v) in self.sparse_repr.to_list()])

        return self._paulis

    @property
    def coeffs(self) -> tuple[float]:
        if len(self._paulis) == 0:
            self._paulis = tuple([Pauli(p) for (p, _) in self.sparse_repr.to_list()])
            self._coeffs = tuple([v for (_, v) in self.sparse_repr.to_list()])

        return self._coeffs


def substitute_parameter(ham: Hamiltonian, para: Parameter, val: float) -> Hamiltonian:
    new_sparse_repr = ham.sparse_repr.assign_parameters({para: val})
    return Hamiltonian(sparse_repr=new_sparse_repr, normalized=ham.normalized)


def sparse_spectral_gap(ham: Hamiltonian) -> float:
    """
    Calculates the spectral gap using the sparse representation of the
    Hamiltonian.
    """
    sparse_mat = ham.sparse_repr.to_matrix(sparse=True)
    eigval, _ = linalg.eigs(sparse_mat, k=2, which="SR")
    return abs(eigval[1] - eigval[0])
