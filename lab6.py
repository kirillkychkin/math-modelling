import numpy as np
from scipy.optimize import fsolve, bisect

# 1. Решение нелинейных уравнений итерационными методами

def f1(x):
    return x**3 - 2*x + 2

def f2(x):
    return np.sin(x) + x - 1

def f3(x):
    return np.log(x) - x**(1/3)

def bisection_method(f, a, b, eps=0.001):
    """
    Метод половинного деления для нахождения корня нелинейного уравнения

    Параметры:
    f - функция, для которой нужно найти корень
    a, b - границы интервала, на котором ищем корень
    eps - точность, которую хотим получить

    Возвращает:
    x_res - найденное приближение корня
    iterations - количество итераций, которое потребовалось для нахождения результата
    """
    iterations = 0
    if f(a) * f(b) >= 0:
        return None, iterations
    
    while (b - a) >= eps:
        iterations += 1
        c = (a + b) / 2
        if abs(f(c)) < eps:
            return c, iterations
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
    
    return (a + b) / 2, iterations


def newton_method(f, f_prime, x0, eps=0.001, max_iter=100):
    """
    Метод Ньютона для нахождения корня нелинейного уравнения

    Параметры:
    f - функция, для которой нужно найти корень
    f_prime - производная функции f
    x0 - начальное приближение
    eps - точность, которую хотим получить
    max_iter - максимальное число итераций

    Возвращает:
    x_res - найденное приближение корня
    iterations - количество итераций, которое потребовалось для нахождения результата
    """
    iterations = 0
    x = x0
    for i in range(max_iter):
        iterations += 1
        x_new = x - f(x) / f_prime(x)
        if abs(x_new - x) < eps:
            return x_new, iterations
        x = x_new
    return x, iterations

# Производные функций
def f1_prime(x):
    return 3*x**2 - 2

def f2_prime(x):
    return np.cos(x) + 1

def f3_prime(x):
    return 1/x - (1/3)*x**(-2/3)

print("1. Решение нелинейных уравнений")

# Уравнение a) x^3 - 2x + 2 = 0
print("\na) x^3 - 2x + 2 = 0")

# Метод половинного деления
root_bisect, iter_bisect = bisection_method(f1, -2, -1)
if root_bisect is not None:
    print(f"Метод половинного деления:")
    print(f"  Корень: {root_bisect:.6f}")
    print(f"  Итераций: {iter_bisect}")
else:
    print("Метод половинного деления: корень не найден на заданном интервале")

# Метод Ньютона
root_newton, iter_newton = newton_method(f1, f1_prime, -1.5)
print(f"Метод Ньютона:")
print(f"  Корень: {root_newton:.6f}")
print(f"  Итераций: {iter_newton}")

# Scipy
try:
    root_scipy = bisect(f1, -2, -1)
    print(f"Scipy bisect: {root_scipy:.6f}")
except:
    print("Scipy bisect: корень не найден")

# Уравнение b) sin(x) + x - 1 = 0
print("\nb) sin(x) + x - 1 = 0")

# Метод половинного деления
root_bisect, iter_bisect = bisection_method(f2, 0, 1)
if root_bisect is not None:
    print(f"Метод половинного деления:")
    print(f"  Корень: {root_bisect:.6f}")
    print(f"  Итераций: {iter_bisect}")
else:
    print("Метод половинного деления: корень не найден на заданном интервале")

# Метод Ньютона
root_newton, iter_newton = newton_method(f2, f2_prime, 0.5)
print(f"Метод Ньютона:")
print(f"  Корень: {root_newton:.6f}")
print(f"  Итераций: {iter_newton}")

# Scipy
try:
    root_scipy = bisect(f2, 0, 1)
    print(f"Scipy bisect: {root_scipy:.6f}")
except:
    print("Scipy bisect: корень не найден")

# Уравнение c) ln(x) - x^(1/3) = 0
print("\nc) ln(x) - x^(1/3) = 0")

# Метод половинного деления
root_bisect, iter_bisect = bisection_method(f3, 1, 10)
if root_bisect is not None:
    print(f"Метод половинного деления:")
    print(f"  Корень: {root_bisect:.6f}")
    print(f"  Итераций: {iter_bisect}")
else:
    print("Метод половинного деления: корень не найден на заданном интервале")

# Метод Ньютона
root_newton, iter_newton = newton_method(f3, f3_prime, 2)
print(f"Метод Ньютона:")
print(f"  Корень: {root_newton:.6f}")
print(f"  Итераций: {iter_newton}")

# Scipy
try:
    root_scipy = bisect(f3, 1, 10)
    print(f"Scipy bisect: {root_scipy:.6f}")
except:
    print("Scipy bisect: корень не найден")

# 2. Решение систем нелинейных уравнений методом Ньютона

print("\n")
print("2. Решение систем нелинейных уравнений")

# Система a)
def system1(vars):
    x, y = vars
    eq1 = np.sin(y) + 2*x - 2
    eq2 = np.cos(x - 1) + y - 0.7
    return [eq1, eq2]

def system1_jacobian(vars):
    x, y = vars
    return [[2, np.cos(y)],
            [-np.sin(x - 1), 1]]

# Метод Ньютона для систем
def newton_system(F, J, x0, eps=0.001, max_iter=100):
    iterations = 0
    x = np.array(x0, dtype=float)
    
    for i in range(max_iter):
        iterations += 1
        F_val = np.array(F(x))
        J_val = np.array(J(x))
        
        try:
            delta = np.linalg.solve(J_val, -F_val)
        except np.linalg.LinAlgError:
            return None, iterations
            
        x_new = x + delta
        
        if np.linalg.norm(delta) < eps:
            return x_new, iterations
        
        x = x_new
    
    return x, iterations

print("\na) Система:")
print("   sin(y) + 2x = 2")
print("   cos(x-1) + y = 0.7")

# Метод Ньютона
solution, iterations = newton_system(system1, system1_jacobian, [0.5, 0.5])
if solution is not None:
    print(f"Метод Ньютона для систем:")
    print(f"  Решение: x = {solution[0]:.6f}, y = {solution[1]:.6f}")
    print(f"  Итераций: {iterations}")
else:
    print("Метод Ньютона для систем: решение не найдено")

# Scipy fsolve
solution_scipy = fsolve(system1, [0.5, 0.5])
print(f"Scipy fsolve: x = {solution_scipy[0]:.6f}, y = {solution_scipy[1]:.6f}")

# Система b)
def system2(vars):
    x, y = vars
    eq1 = np.sin(x + 1) - y - 1.2
    eq2 = 2*x + np.cos(y) - 2
    return [eq1, eq2]

def system2_jacobian(vars):
    x, y = vars
    return [[np.cos(x + 1), -1],
            [2, -np.sin(y)]]

print("\nb) Система:")
print("   sin(x+1) - y = 1.2")
print("   2x + cos(y) = 2")

# Метод Ньютона
solution, iterations = newton_system(system2, system2_jacobian, [0.5, -0.5])
if solution is not None:
    print(f"Метод Ньютона для систем:")
    print(f"  Решение: x = {solution[0]:.6f}, y = {solution[1]:.6f}")
    print(f"  Итераций: {iterations}")
else:
    print("Метод Ньютона для систем: решение не найдено")

# Scipy fsolve
solution_scipy = fsolve(system2, [0.5, -0.5])
print(f"Scipy fsolve: x = {solution_scipy[0]:.6f}, y = {solution_scipy[1]:.6f}")
