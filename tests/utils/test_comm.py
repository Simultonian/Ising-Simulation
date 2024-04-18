from qiskit.quantum_info import SparsePauliOp, Pauli

x, y = Pauli("X"), Pauli("Y")
ca, cb = 1, 2
ham = SparsePauliOp([x, y], [1.0, 2.0])

ta = (ca * cb), (x @ y)
tb = (-cb * ca), (y @ x)

print(ta, tb)
print(SparsePauliOp([ta[1], tb[1]], [ta[0], tb[0]]))
