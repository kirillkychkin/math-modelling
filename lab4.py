import numpy as np
from scipy.linalg import lu, qr, cholesky, solve_triangular

def solve_from_lu(P, L, U, b):
    y = solve_triangular(L, P @ b, lower=True)
    x = solve_triangular(U, y)
    return x

def solve_from_qr(Q, R, b):
    y = Q.T @ b
    x = solve_triangular(R, y)
    return x

def solve_from_chol(L, b):
    y = solve_triangular(L, b, lower=True)
    x = solve_triangular(L.T, y)
    return x

def thomas_algorithm(a, b, c, d):
    """
    a: поддиагональ (n-1)
    b: главная диагональ (n)
    c: наддиагональ (n-1)
    d: правая часть (n)
    """
    n = len(b)
    c_ = np.zeros(n-1)
    d_ = np.zeros(n)

    c_[0] = c[0] / b[0]
    d_[0] = d[0] / b[0]

    for i in range(1, n-1):
        denom = b[i] - a[i-1] * c_[i-1]
        c_[i] = c[i] / denom
        d_[i] = (d[i] - a[i-1]*d_[i-1]) / denom

    d_[n-1] = (d[n-1] - a[n-2]*d_[n-2]) / (b[n-1] - a[n-2]*c_[n-2])

    x = np.zeros(n)
    x[-1] = d_[n-1]
    for i in reversed(range(n-1)):
        x[i] = d_[i] - c_[i]*x[i+1]

    return x

# Решения
A_a = np.array([
    [7, 1, 1, 0],
    [1, 5, 2, 1],
    [2, 3, -3, 3],
    [3, 4, 5, 5]
], dtype=float)
b_a = np.array([7, 0, -1, -2], dtype=float)

P, L, U = lu(A_a)
x_a_lu = solve_from_lu(P, L, U, b_a)

Q, R = qr(A_a)
x_a_qr = solve_from_qr(Q, R, b_a)

print("Задача (a)")
print("LU:", x_a_lu)
print("QR:", x_a_qr)
print()

A_b = np.array([
    [4, -6, 0, 8, 0, 0],
    [-6, 8, -12, 0, 16, 0],
    [0, -12, 16, 0, 0, 10],
    [8, 0, 0, 20, -8, 0],
    [0, 16, 0, -8, 24, -8],
    [0, 0, 10, 0, -8, 28]
], dtype=float)
b_b = np.array([24, 54, 84, 48, 72, 158], dtype=float)

P, L, U = lu(A_b)
x_b_lu = solve_from_lu(P, L, U, b_b)

Q, R = qr(A_b)
x_b_qr = solve_from_qr(Q, R, b_b)

print("Задача (б)")
print("LU:", x_b_lu)
print("QR:", x_b_qr)

try:
    L = cholesky(A_b, lower=True)
    x_b_chol = solve_from_chol(L, b_b)
    print("Cholesky:", x_b_chol)
except Exception as e:
    print("Cholesky: не существует (матрица не положительно определена)")
print()

n = 6 
A_v = 2*np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
b_v = np.array([3, -4, 4, -4, 4, -3], dtype=float)

a = -np.ones(n-1)      # поддиагональ
b_diag = 2*np.ones(n)  # главная диагональ
c = -np.ones(n-1)      # наддиагональ

# метод прогонки
x_v_thomas = thomas_algorithm(a, b_diag, c, b_v)

# LU и QR для проверки
P, L, U = lu(A_v)
x_v_lu = solve_from_lu(P, L, U, b_v)

Q, R = qr(A_v)
x_v_qr = solve_from_qr(Q, R, b_v)

print("Задача (в)")
print("Прогонка:", x_v_thomas)
print("LU:", x_v_lu)
print("QR:", x_v_qr)

L = cholesky(A_v, lower=True)
x_v_chol = solve_from_chol(L, b_v)
print("Cholesky:", x_v_chol)
print()

A_g = np.array([
    [10, -1, 0, 0, 0, 0],
    [1, -10, 3, 0, 0, 0],
    [0, 1, -10, -2, 0, 0],
    [0, 0, 2, -10, -1, 0],
    [0, 0, 0, 1, -10, 1],
    [0, 0, 0, 0, 2, -10]
], dtype=float)
b_g = np.array([10, 4, -10, 1, -10, 2], dtype=float)

P, L, U = lu(A_g)
x_g_lu = solve_from_lu(P, L, U, b_g)

Q, R = qr(A_g)
x_g_qr = solve_from_qr(Q, R, b_g)

print("Задача (г)")
print("LU:", x_g_lu)
print("QR:", x_g_qr)

try:
    L = cholesky(A_g, lower=True)
    x_g_chol = solve_from_chol(L, b_g)
    print("Cholesky:", x_g_chol)
except Exception:
    print("Cholesky: не существует (матрица не положительно определена)")
