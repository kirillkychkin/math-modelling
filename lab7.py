import numpy as np
from scipy.linalg import solve, lstsq
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

print("=" * 60)
print("ПРАКТИЧЕСКАЯ РАБОТА №7")
print("Аппроксимация данных методом наименьших квадратов")
print("=" * 60)

# ЗАДАНИЕ 1
print("\n" + "="*50)
print("ЗАДАНИЕ 1")
print("="*50)

# Данные
x = np.array([-2, -1, 2, 5, 7])
y = np.array([1.3, 0.8, 0, 1.5, 1.2])

# 1. Параболическая аппроксимация (вручную)
print("\n1. ПАРАБОЛИЧЕСКАЯ АППРОКСИМАЦИЯ (ручной расчет):")

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
print(f"c0 = {c0:.6f}, c1 = {c1:.6f}, c2 = {c2:.6f}")
print(f"Уравнение: y = {c0:.6f} + {c1:.6f}x + {c2:.6f}x²")

# 2. Параболическая аппроксимация (scipy.linalg.lstsq)
print("\n2. ПАРАБОЛИЧЕСКАЯ АППРОКСИМАЦИЯ (scipy.linalg.lstsq):")

A_lstsq = np.column_stack([np.ones_like(x), x, x**2])
coefficients, residuals, rank, s = lstsq(A_lstsq, y, cond=None)
c0_lstsq, c1_lstsq, c2_lstsq = coefficients

print(f"c0 = {c0_lstsq:.6f}, c1 = {c1_lstsq:.6f}, c2 = {c2_lstsq:.6f}")
print(f"Сумма квадратов отклонений: {residuals:.6f}")

# 3. Логарифмическая аппроксимация
print("\n3. ЛОГАРИФМИЧЕСКАЯ АППРОКСИМАЦИЯ:")

def log_func(x, a, b):
    return a + b * np.log(x)

x_pos = x[x > 0]
y_pos = y[x > 0]

if len(x_pos) > 0:
    popt_log, pcov_log = curve_fit(log_func, x_pos, y_pos)
    a_log, b_log = popt_log
    
    y_log_pred = log_func(x_pos, a_log, b_log)
    ssr_log = np.sum((y_pos - y_log_pred)**2)
    
    print(f"a = {a_log:.6f}, b = {b_log:.6f}")
    print(f"Уравнение: y = {a_log:.6f} + {b_log:.6f}·ln(x)")
    print(f"Сумма квадратов отклонений: {ssr_log:.6f}")
else:
    print("Нет положительных значений x для логарифмической аппроксимации")

# График для задания 1
plt.figure(figsize=(12, 8))
plt.scatter(x, y, color='red', s=50, label='Исходные данные', zorder=5)

x_fine = np.linspace(min(x)-0.5, max(x)+0.5, 100)
y_parabola = c0_lstsq + c1_lstsq * x_fine + c2_lstsq * x_fine**2
plt.plot(x_fine, y_parabola, 'b-', linewidth=2, 
         label=f'Парабола: y = {c0_lstsq:.3f} + {c1_lstsq:.3f}x + {c2_lstsq:.3f}x²')

if len(x_pos) > 0:
    x_log_fine = np.linspace(min(x_pos), max(x_pos), 100)
    y_log = log_func(x_log_fine, a_log, b_log)
    plt.plot(x_log_fine, y_log, 'g-', linewidth=2, 
             label=f'Логарифм: y = {a_log:.3f} + {b_log:.3f}·ln(x)')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Задание 1: Аппроксимация данных')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ЗАДАНИЕ 2
print("\n" + "="*50)
print("ЗАДАНИЕ 2")
print("="*50)

# Данные о прибыли
years = np.array([1, 2, 3, 4, 5, 6])
profit = np.array([3.9, 4.3, 2.5, 2.75, 3.4, 3.6])

def poly_func(x, a, b, c):
    return a * x**2 + b * x + c

popt_profit, pcov_profit = curve_fit(poly_func, years, profit)
a_profit, b_profit, c_profit = popt_profit

print("Аппроксимация данных о прибыли:")
print(f"Уравнение: y = {a_profit:.6f}x² + {b_profit:.6f}x + {c_profit:.6f}")

# Прогноз
year_7, year_8 = 7, 8
profit_7 = poly_func(year_7, a_profit, b_profit, c_profit)
profit_8 = poly_func(year_8, a_profit, b_profit, c_profit)

print(f"Ожидаемая прибыль на 7-й год: {profit_7:.3f}")
print(f"Ожидаемая прибыль на 8-й год: {profit_8:.3f}")

y_profit_pred = poly_func(years, a_profit, b_profit, c_profit)
ssr_profit = np.sum((profit - y_profit_pred)**2)
print(f"Сумма квадратов отклонений: {ssr_profit:.6f}")

# График для задания 2
plt.figure(figsize=(10, 6))
plt.scatter(years, profit, color='red', s=50, label='Фактическая прибыль', zorder=5)

years_fine = np.linspace(0.5, 8.5, 100)
profit_fine = poly_func(years_fine, a_profit, b_profit, c_profit)
plt.plot(years_fine, profit_fine, 'b-', linewidth=2, label='Аппроксимирующая кривая')

plt.scatter([year_7, year_8], [profit_7, profit_8], color='green', s=60, 
           label='Прогнозная прибыль', zorder=5)

plt.xlabel('Год')
plt.ylabel('Прибыль')
plt.title('Задание 2: Аппроксимация данных о прибыли фирмы')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\n" + "="*60)
print("РАБОТА ЗАВЕРШЕНА")
print("="*60)