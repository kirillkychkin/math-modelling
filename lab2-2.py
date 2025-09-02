import numpy as np

A1 = np.eye(3)[:, ::-1]

B1 = np.matrix([[0, 2, -3], [-2, 0, 6], [3, -6, 0]])

plus1 = np.add(A1, B1)
print("A1 + B1 =\n", plus1)

multiply11 = np.matmul(A1, B1)
print("A1 * B1 =\n", multiply11)

multiply12 = np.matmul(B1, A1)
print("B1 * A1 =\n", multiply12)

A2 = np.array([[1, 3 + 2i],
    [3- 2i, 5]])
print("A2 =\n", A2)