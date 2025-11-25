import numpy as np
import matplotlib.pyplot as plt
from methods import euler_method, modified_euler_method, runge_kutta_method

def exact_solution(x):
    return np.exp(x**2 / 2)

def f(x, y):
    return x * y

# Параметры
x0 = 0
y0 = 1
h = 0.1
n = 10
x_end = x0 + n * h

# Сетка
x = np.linspace(x0, x_end, n + 1)

# Вычисление решений
y_euler = euler_method(x, y0, n, h, f)
y_modified = modified_euler_method(x, y0, n, h, f)
y_rk = runge_kutta_method(x, y0, n, h, f)
y_exact = exact_solution(x)

# Вывод результатов в таблицу
print("x\t\tЭйлер\t\tМод.Эйлер\tРунге-Кутта\tТочное")
for i in range(n + 1):
    print(f"{x[i]:.1f}\t\t{y_euler[i]:.6f}\t{y_modified[i]:.6f}\t{y_rk[i]:.6f}\t{y_exact[i]:.6f}")

# Построение графиков
plt.figure(figsize=(12, 8))

plt.plot(x, y_exact, '*--', linewidth=2, label='Точное решение')
plt.plot(x, y_euler, 'o--',  label='Метод Эйлера', color='red')
plt.plot(x, y_modified, 'bs--', label='Модифицированный Эйлер')
plt.plot(x, y_rk, 'g^--', label='Рунге-Кутта 4-го порядка')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.title('Сравнение численных методов решения задачи Коши')
plt.legend()
plt.grid(True)

plt.show()