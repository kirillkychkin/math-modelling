from scipy.linalg import norm
import numpy as np

low_value = -10
high_value = 10
n = 4
x = np.random.uniform(low=low_value, high=high_value, size=n)
A = np.random.uniform(low=low_value, high=high_value, size=(n, n))
print("vector x:",x)
print("matrix A:\n", A)

def manhattan_norm(x):
    return np.sum(np.abs(x))

def euclidean_norm(x):
    return np.sqrt(np.sum(x**2))

def chebyshev_norm(x):
    return np.max(np.abs(x))

def matrix_l1_norm(A):
    return np.max(np.sum(np.abs(A), axis=0))

def frobenius_norm(A):
    return np.sqrt(np.sum(A**2))

def inf_norm(A):
    return np.max(np.sum(np.abs(A), axis=1))

def experiment(n_values, num_trials):
    results = []
    
    for n in n_values:
        for _ in range(num_trials):
            # Генерация случайной матрицы
            A = np.random.randn(n, n)
            
            # Вычисление норм
            norm_1 = matrix_l1_norm(A)
            norm_inf = inf_norm(A)
            norm_E = frobenius_norm(A)
            
            # Проверка неравенства для α = 1
            left_bound_1 = norm_1 / np.sqrt(n)
            right_bound_1 = norm_E
            
            # Проверка неравенства для α = ∞
            left_bound_inf = norm_E
            right_bound_inf = norm_inf * np.sqrt(n)
            
            results.append({
                'n': n,
                'norm_1': norm_1,
                'norm_inf': norm_inf,
                'norm_E': norm_E,
                'left_bound_1': left_bound_1,
                'right_bound_1': right_bound_1,
                'left_bound_inf': left_bound_inf,
                'right_bound_inf': right_bound_inf,
                'valid_1': left_bound_1 <= right_bound_1,
                'valid_inf': left_bound_inf <= right_bound_inf
            })
    
    return results

def analyze_results(results):
    n_values = sorted(set(r['n'] for r in results))
    
    print("Результаты эксперимента:")
    
    for n in n_values:
        n_results = [r for r in results if r['n'] == n]
        
        valid_1_count = sum(1 for r in n_results if r['valid_1'])
        valid_inf_count = sum(1 for r in n_results if r['valid_inf'])
        
        valid_1_percent = valid_1_count / len(n_results) * 100
        valid_inf_percent = valid_inf_count / len(n_results) * 100
        
        print(f"n = {n}:")
        print(f"  Неравенство для α=1 выполняется в {valid_1_percent:.2f}% случаев")
        print(f"  Неравенство для α=∞ выполняется в {valid_inf_percent:.2f}% случаев")

print("Manhattan norm (self):", manhattan_norm(x))
print("Manhattan norm (scipy.linalg.norm):", norm(x, ord=1))
print("Euclidean norm (self):", euclidean_norm(x))
print("Euclidean norm (scipy.linalg.norm):", norm(x))
print("Chebyshev norm (self):", chebyshev_norm(x))
print("Chebyshev norm (scipy.linalg.norm):", norm(x, ord=np.inf))

print("L1 norm (self):", matrix_l1_norm(A))
print("L1 norm (scipy.linalg.norm):", norm(A, ord=1))
print("Frobenius norm (self):", frobenius_norm(A))
print("Frobenius norm (scipy.linalg.norm):", norm(A))
print("Infinity norm (self):", inf_norm(A))
print("Infinity norm (scipy.linalg.norm):", norm(A, ord=np.inf))

n_values = [5, 10, 20, 50, 100, 250, 500, 1000]
num_trials = 30
results = experiment(n_values, num_trials)
analyze_results(results)