import numpy as np

n = [5, 10, 20, 30, 100, 500]
low_value = -100
high_value = 100
matrices = []

def min_matrix(n):
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A[i, j] = min(i + 1, j + 1) 
    return A

for i in n:
    matrices.append(min_matrix(i))
    print("\n")
    print("=" * 40)
    print("При n = ", i)
    print("Матрица A:\n", matrices[-1])
    
    cond_1 = np.linalg.cond(matrices[-1], 1)
    cond_inf = np.linalg.cond(matrices[-1], np.inf)
    cond_fro = np.linalg.cond(matrices[-1], 'fro')

    print(f"cond 1(A) = {cond_1:.6f}")
    print(f"cond inf(A) = {cond_inf:.6f}")
    print(f"cond fro(A) = {cond_fro:.6f}")
