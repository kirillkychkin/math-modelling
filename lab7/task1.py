import numpy as np
from scipy.linalg import solve, lstsq
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def parabolic_approximation(x, y):
    print("\n1. Параболическая аппроксимация (ручной расчет):")

    n = len(x)
    sum_x = np.sum(x)
    sum_x2 = np.sum(x**2)
    sum_x3 = np.sum(x**3)
    sum_x4 = np.sum(x**4)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2y = np.sum(x**2 * y)

    A = np.array([
        [n, sum_x, sum_x2],
        [sum_x, sum_x2, sum_x3],
        [sum_x2, sum_x3, sum_x4]
    ])
    b = np.array([sum_y, sum_xy, sum_x2y])

    c0, c1, c2 = solve(A, b)
    
    print(f"c0 = {c0}, c1 = {c1}, c2 = {c2}")
    print(f"Уравнение: y = {c0} + {c1}x + {c2}x²")
    return c0, c1, c2

def parabolic_approximation_scipy(x,y):
    print("\n2. Параболическая аппроксимация (scipy.linalg.lstsq):")
    A_lstsq = np.column_stack([np.ones_like(x), x, x**2])
    coefficients, residuals, rank, s = lstsq(A_lstsq, y, cond=None)
    c0_lstsq, c1_lstsq, c2_lstsq = coefficients

    print(f"c0 = {c0_lstsq}, c1 = {c1_lstsq}, c2 = {c2_lstsq}")
    print(f"Сумма квадратов отклонений: {residuals}")

    return c0_lstsq, c1_lstsq, c2_lstsq

def log_func(x, a, b):
    return a + b * np.log(x)
def logariphmic_approximation(x, y):
    print("\n3. Логарифмическая аппроксимация:")

    x_pos = x[x > 0]
    y_pos = y[x > 0]

    if len(x_pos) > 0:
        popt_log, pcov_log = curve_fit(log_func, x_pos, y_pos)
        a_log, b_log = popt_log
        
        y_log_pred = log_func(x_pos, a_log, b_log)
        ssr_log = np.sum((y_pos - y_log_pred)**2)
        
        print(f"a = {a_log}, b = {b_log}")
        print(f"Уравнение: y = {a_log} + {b_log}·ln(x)")
        print(f"Сумма квадратов отклонений: {ssr_log}")

        return x_pos, y_pos, a_log, b_log
    else:
        print("Нет положительных значений x для логарифмической аппроксимации")
    

def solveTaskOne():
    print("Задание 1")

    # Данные
    x = np.array([-2, -1, 2, 5, 7])
    y = np.array([1.3, 0.8, 0, 1.5, 1.2])

    c0, c1, c2 = parabolic_approximation(x, y)

    c0_lstsq, c1_lstsq, c2_lstsq = parabolic_approximation_scipy(x, y)

    x_pos, y_pos, a_log, b_log = logariphmic_approximation(x, y)

    # График для задания 1
    plt.figure(figsize=(12, 8))
    plt.scatter(x, y, color='red', s=50, label='Исходные данные', zorder=5)

    x_fine = np.linspace(min(x)-0.5, max(x)+0.5, 100)
    y_parabola = c0_lstsq + c1_lstsq * x_fine + c2_lstsq * x_fine**2
    plt.plot(x_fine, y_parabola, 'b-', linewidth=2, 
            label=f'Парабола: y = {c0_lstsq} + {c1_lstsq}x + {c2_lstsq}x²')

    if len(x_pos) > 0:
        x_log_fine = np.linspace(min(x_pos), max(x_pos), 100)
        y_log = log_func(x_log_fine, a_log, b_log)
        plt.plot(x_log_fine, y_log, 'g-', linewidth=2, 
                label=f'Логарифм: y = {a_log} + {b_log}·ln(x)')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Задание 1: Аппроксимация данных')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
