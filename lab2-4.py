import numpy as np

A = np.array([[9, 2, 3, 5],
              [3, 1, 3, 5],
              [1, 1, 4, 2],
              [1, 2, 4, 3]])

B = np.array([[4, 1, 0, 0, 0],
              [3, 4, 1, 0, 0],
              [0, 3, 4, 1, 0],
              [0, 0, 3, 4, 1],
              [0, 0, 0, 3, 4]])

from numpy import linalg as LA

eigenvalues, eigenvectors = LA.eig(A)
print(eigenvalues)
print(eigenvectors)

eigenvalues, eigenvectors = LA.eig(B)
print(eigenvalues)
print(eigenvectors)