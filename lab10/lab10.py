import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

def task1():
    a, b = 0.9, 1.2
    h = 0.1
    epsilon = 1e-3
    
    x = np.arange(a, b + h/2, h)
    n = len(x)
    
    A = np.zeros((n, n))
    b_vec = np.zeros(n)
    
    for i in range(n):
        if i == 0: 
            A[i, i] = 1 - 0.5/(2*h)
            if i+1 < n:
                A[i, i+1] = 0.5/(2*h)
            b_vec[i] = 2
        
        elif i == n-1:
            A[i, i] = 1
            b_vec[i] = 1
        
        else:
            A[i, i-1] = 1/(h**2) - x[i]/(2*h)
            A[i, i] = -2/(h**2) + 2
            A[i, i+1] = 1/(h**2) + x[i]/(2*h)
            
            b_vec[i] = x[i] + 1
    
    y = solve(A, b_vec)
    
    print(f"Диапазон: [{a}, {b}], шаг h = {h}")
    print("Узлы и приближенные решения:")
    for i in range(n):
        print(f"x = {x[i]:.2f}, y = {y[i]:.6f}")
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(x, y, 'bo-', linewidth=2, markersize=6, label='Приближенное решение')
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.title('Задача 1: Приближенное решение')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    return x, y

def task2():
    def exact_solution(x):
        return np.cos(np.pi * x) / (np.pi**2) + x - 1
    
    results = {}
    
    for N in [10, 20]:
        a, b = 0, 1
        h = 1/N
        
        x = np.linspace(a, b, N+1)
        n = len(x)
        
        A = np.zeros((n, n))
        b_vec = np.zeros(n)
        
        for i in range(n):
            if i == 0:
                A[i, i] = -1/h
                A[i, i+1] = 1/h
                b_vec[i] = 1
            
            elif i == n-1:
                A[i, i] = 1
                b_vec[i] = -1/(np.pi**2)
            
            else:
                A[i, i-1] = 1/(h**2)
                A[i, i] = -2/(h**2)
                A[i, i+1] = 1/(h**2)
                b_vec[i] = -np.cos(np.pi * x[i])
        
        u = solve(A, b_vec)
        
        u_exact = exact_solution(x)
        
        error = np.abs(u - u_exact)
        max_error = np.max(error)
        
        results[N] = {
            'x': x,
            'u': u,
            'u_exact': u_exact,
            'error': error,
            'max_error': max_error,
            'h': h
        }
        
        print(f"Шаг h = {h:.3f}")
        print(f"Максимальная погрешность: {max_error:.6f}")
        print("Первые 5 узлов:")
        for j in range(min(5, n)):
            print(f"x = {x[j]:.3f}, u_num = {u[j]:.6f}, u_exact = {u_exact[j]:.6f}, error = {error[j]:.6e}")
    
    plt.subplot(1, 2, 2)
    
    colors = ['b', 'r']
    for idx, (N, res) in enumerate(results.items()):
        plt.plot(res['x'], res['u'], colors[idx]+'o-', linewidth=1, markersize=4, 
                label=f'N={N}, h={1/N:.3f}')
    
    x_fine = np.linspace(0, 1, 100)
    u_exact_fine = exact_solution(x_fine)
    plt.plot(x_fine, u_exact_fine, 'k--', linewidth=2, label='Точное решение')
    
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('Задача 2: Сравнение решений')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.figure(figsize=(8, 5))
    for idx, (N, res) in enumerate(results.items()):
        plt.plot(res['x'], res['error'], colors[idx]+'o-', linewidth=1, markersize=4, 
                label=f'N={N}, max_err={res["max_error"]:.3e}')
    
    plt.xlabel('x')
    plt.ylabel('Погрешность')
    plt.title('Задача 2: Погрешность численного решения')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    return results

def main():
    
    x1, y1 = task1()
    results2 = task2()
    
    plt.tight_layout()
    plt.show()
    
    # Проверка аналитического решения в граничных точках
    x_test = np.array([0, 1])
    u_test = np.cos(np.pi * x_test) / (np.pi**2) + x_test - 1
    u_prime_0 = -np.pi*np.sin(np.pi*0)/(np.pi**2) + 1  # u'(0) = 1
    
    print(f"u(0) аналитически = {u_test[0]:.6f}, u'(0) = {u_prime_0:.6f}")
    print(f"u(1) аналитически = {u_test[1]:.6f}, задано: -1/π² = {-1/(np.pi**2):.6f}")

if __name__ == "__main__":
    main()