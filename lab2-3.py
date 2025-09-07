import numpy as np
import pandas as pd

D = np.array([[1 + 1j, 3, 2],
              [8, -5j, 7 - 1j]])
F = np.array([[2 + 1j, 1, 0, 0],
              [1, 2 - 1j, 0, 0],
              [0, 0, 1 + 1j, 1],
              [0, 0, 1, 1 - 1j]])

d_conjugate = np.conjugate(D).T
f_conjugate = np.conjugate(F).T

print("D^* =\n", pd.DataFrame(d_conjugate))
print("F^* =\n", pd.DataFrame(f_conjugate))