import numpy as np
import pandas as pd

def add(A: np.ndarray, B: np.ndarray):
    if(A.shape != B.shape):
        print("Can't add matrices with different sizes")
    else:
        print("A + B =\n", pd.DataFrame(np.add(A, B)))

def multiply(A: np.ndarray, B: np.ndarray):
    if(A.shape[1] != B.shape[0]):
        print("Can't multiply A * B")
    else:
        print("A * B =\n", pd.DataFrame(np.matmul(A, B)))
    
    if(A.shape[0] != B.shape[1]):
        print("Can't multiply B * A")
    else:
        print("B * A =\n", pd.DataFrame(np.matmul(B, A)))

A1 = np.eye(3)[:, ::-1]
B1 = np.matrix([[0, 2, -3], 
                [-2, 0, 6], 
                [3, -6, 0]])
print("task a)")
add(A1, B1)
multiply(A1, B1)

A2 = np.array([[1, 3 + 2j], 
               [3 - 2j, 5]])
B2 = np.array([[3, 2, 1],
               [7, 1, 4]])

print("task b)")
add(A2, B2)
multiply(A2, B2)

A3 = np.array([[0, 1, 3],
               [1, 5, 6]])
B3 = np.array([[1, 4, 9],
               [0, 1, 6]])

print("task c)")
add(A3, B3)
multiply(A3, B3)