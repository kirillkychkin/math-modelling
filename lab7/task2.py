import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def solveTaskTwo():
    print("Задание 2")

    # Данные о прибыли
    years = np.array([1, 2, 3, 4, 5, 6])
    profit = np.array([3.9, 4.3, 2.5, 2.75, 3.4, 3.6])

    def poly_func(x, a, b, c):
        return a * x**2 + b * x + c

    popt_profit, pcov_profit = curve_fit(poly_func, years, profit)
    a_profit, b_profit, c_profit = popt_profit

    print("Аппроксимация данных о прибыли:")
    print(f"Уравнение: y = {a_profit}x² + {b_profit}x + {c_profit}")

    # Прогноз
    year_7, year_8 = 7, 8
    profit_7 = poly_func(year_7, a_profit, b_profit, c_profit)
    profit_8 = poly_func(year_8, a_profit, b_profit, c_profit)

    print(f"Ожидаемая прибыль на 7-й год: {profit_7}")
    print(f"Ожидаемая прибыль на 8-й год: {profit_8}")

    y_profit_pred = poly_func(years, a_profit, b_profit, c_profit)
    ssr_profit = np.sum((profit - y_profit_pred)**2)
    print(f"Сумма квадратов отклонений: {ssr_profit}")

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