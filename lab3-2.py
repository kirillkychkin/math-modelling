import numpy as np
from scipy.linalg import norm

# Определение систем уравнений
A = np.array([[8, 5, 2],
            [21, 19, 16],
            [39, 48, 53]])

b1 = np.array([14, 56, 140])

B = np.array([[8, 5, 2],
            [21, 19, 16],
            [39, 48, 53]])

b2 = np.array([15, 56, 140])

C = np.array([[6, 3, 1],
            [6, 9, 2],
            [3, 11, 15]])

b3 = np.array([10, 17, 29])

D = np.array([[6, 3, 1],
            [6, 9, 2],
            [3, 11, 15]])

b4 = np.array([10, 17, 28])

# Решение систем уравнений
def solve_linear_systems():
    systems = [
        {'matrix': A, 'vector': b1, 'name': 'Система a'},
        {'matrix': B, 'vector': b2, 'name': 'Система b'},
        {'matrix': C, 'vector': b3, 'name': 'Система c'},
        {'matrix': D, 'vector': b4, 'name': 'Система d'}
    ]
    
    solutions = {}
    
    for system in systems:
        print(f"\n{system['name']}")
        print("-" * 40)

        # Решение системы
        x = np.linalg.solve(system['matrix'], system['vector'])
        solutions[system['name']] = x
        
        print(f"Решение:")
        print(f"x = {x[0]:.8f}")
        print(f"y = {x[1]:.8f}")
        print(f"z = {x[2]:.8f}")
            
    return solutions

# Вычисление числа обусловленности
def compute_condition_numbers():

    matrices = [
        {'matrix': A, 'name': 'Матрица A'},
        {'matrix': B, 'name': 'Матрица B'},
        {'matrix': C, 'name': 'Матрица C'},
        {'matrix': D, 'name': 'Матрица D'}
    ]
    
    for mat in matrices:
        print(f"\n{mat['name']}")
        print("-" * 40)
        
        A_mat = mat['matrix']
        A_inv = np.linalg.inv(A_mat)
        
        # Наша реализация
        cond_1 = norm(A_mat, ord=1) * norm(A_inv, ord=1)
        cond_E = norm(A_mat) * norm(A_inv)
        cond_inf = norm(A_mat, ord=np.inf) * norm(A_inv, ord=np.inf)
        
        print("Наша реализация:")
        print(f"cond 1(A) = {cond_1:.6f}")
        print(f"cond E(A) = {cond_E:.6f}")
        print(f"cond inf(A) = {cond_inf:.6f}")
        
        # NumPy реализация
        cond_numpy_1 = np.linalg.cond(A_mat, 1)
        cond_numpy_inf = np.linalg.cond(A_mat, np.inf)
        cond_numpy_fro = np.linalg.cond(A_mat, 'fro')
        
        print("\nNumPy реализация:")
        print(f"cond(A, 1) = {cond_numpy_1:.6f}")
        print(f"cond(A, 'fro') = {cond_numpy_fro:.6f}")
        print(f"cond(A, inf) = {cond_numpy_inf:.6f}")

# Анализ влияния малых изменений
def analyze_sensitivity():
    # Анализ для систем a и b
    print("\n1. Сравнение систем a и b:")
    print("-" * 30)
    
    x_a = np.linalg.solve(A, b1)
    x_b = np.linalg.solve(B, b2)
    
    delta_b = b2 - b1
    delta_x = x_b - x_a
    
    cond_2 = np.linalg.cond(A, 2)
    
    print(f"Изменение в правой части: {delta_b}")
    
    print(f"\nИзменение в решении: {delta_x}")
    
    print(f"\nЧисло обусловленности cond E(A) = {cond_2:.6f}")
    
    # Анализ для систем c и d
    print("\n2. Сравнение систем c и d:")
    print("-" * 30)
    
    x_c = np.linalg.solve(C, b3)
    x_d = np.linalg.solve(D, b4)
    
    delta_b_cd = b4 - b3
    delta_x_cd = x_d - x_c
    
    cond_2_cd = np.linalg.cond(C, 2)
    
    print(f"Изменение в правой части: {delta_b_cd}")
    
    print(f"\nИзменение в решении: {delta_x_cd}")
    
    print(f"\nЧисло обусловленности cond E(C) = {cond_2_cd:.6f}")

solutions = solve_linear_systems()
compute_condition_numbers()
analyze_sensitivity()