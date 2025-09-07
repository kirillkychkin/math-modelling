import numpy as np
from numpy.linalg import det, inv, solve

m = 3

A = np.random.randint(-9, 10, size=(m, m)).astype(float)
b = np.ones(m, dtype=float)

print("Matrix A:\n", A)
detA = det(A)
print("Determinant of A:", detA)

inverse = inv(A)
print("Inverse of A:\n", inverse)

y_via_inv = np.dot(inverse, b)
y_via_solve = solve(A, b)

print("Solution y via inverse:\n", y_via_inv)
print("\nSolution y via solve(A, b):\n", y_via_solve)
