import numpy as np
import math
# исходные данные
A = [[5, 1, 1], [2, 6, -1], [2, 3, 10]]
n = 3
b = [11, 13, 18]
def parameterized_b(b, n):
    new_b = [b[0] + 0.8 * n, b[1] + 0.9 * n, b[2] - 0.5 * n]
    return new_b

from convergence import isDiagDominant
'''
В СЛАУ вида:`
a11 * x1 + a12 * x2 + a13 * x3 = b1
a21 * x1 + a22 * x2 + a23 * x3 = b2
a31 * x1 + a32 * x2 + a33 * x3 = b3
....
решения уравнений можно выразить следующим образом:
x1 = (1 / a11) * (b1 - a12 * x2 - a13 * x3)
x2 = (1 / a22) * (b2 - a21 * x1 - a23 * x3)
x3 = (1 / a33) * (b3 - a31 * x1 - a32 * x2)
....

тогда формулы для метода якоби:
x1 ^(k + 1) = (1 / a11) * (b1 - a12 * x2 ^ (k) - a13 * x3 ^ (k))
x2 ^(k + 1) = (1 / a22) * (b2 - a21 * x1 ^ (k) - a23 * x3 ^ (k))
x3 ^(k + 1) = (1 / a33) * (b3 - a31 * x1 ^ (k) - a32 * x2 ^ (k))

тогда формулы для метода зейделя:
x1 ^(k + 1) = (1 / a11) * (b1 - a12 * x2 ^ (k) - a13 * x3 ^ (k))
x2 ^(k + 1) = (1 / a22) * (b2 - a21 * x1 ^ (k + 1) - a23 * x3 ^ (k))
x3 ^(k + 1) = (1 / a33) * (b3 - a31 * x1 ^ (k+ 1) - a32 * x2 ^ (k + 1))
.....
k - метод итерации
'''

def errorCalc(x, xn):
    max_err = 0
    # по каждому решению ищем максимальное различие между соответствующими решениями x между текущим и прошлым шагом, знак роли не играет
    for i in range(len(x)):
        err = abs(x[i] - xn[i])
        if err > max_err:
            max_err = err
    # максимальное различие и есть ошибка - возвращем её
    return max_err

# Пример действия 
# вычисляет всю сумму решения, которую нужно отнять от b
# a11 * x1 + a12 * x2 + a13 * x3
# поэтому в решении в конце отнимаем диагональную часть a11 * x1
def fullBracket(A, x, i):
    res = 0
    for k in range(len(A)):
        res += A[i][k] * x[k]
    return res

# eps - точность, которую хотим получить
def zeidel(A, b, eps):
    # первичные значения, с которых начинаем вычисления
    start = [0] * len(A)

    # решения x по всем шагам
    X_res = []
    # пихаем первичные значения (чтобы начать вычисления с них)
    X_res.append(start)
    # задаем максимальную ошибку
    error = float('inf')
    # решение текущего шага
    xn = [0] * len(A)
    # номер итерации
    k = 0
    # совершаем итерации пока не достигнем желаемой точности
    while error > eps:
        # инкрементируем итерацию (нулевая итерация уже сделана, это первичные значения)
        k += 1
        # получаем значения решений с последней итерации
        # используем slice, т.к. прямое присвоение копирует
        # не значение массива, а ссылку на него
        # тогда как slice генерирует новый объект
        x = X_res[-1][:]
        # текущие значения для первого значения решения равны предыдушему шагу
        xc = x[:]
        # вычисляем значения текущей итерации по формуле
        for i in range(len(A)):
            # опираемся на текущие значения
            xn[i] = (1 / float(A[i][i])) * (b[i] - (fullBracket(A, xc, i) - A[i][i] * xc[i]))
            # записываем текущее значения для дальнейших решений
            xc[i] = xn[i]
        # вставляем текущий шаг в список шагов
        X_res.append(xn[:])
        # вычисляем текущую ошибку
        error = errorCalc(x, xn)
        # если никак не можем подобраться к решению, ошибка скачет и прошло много итераций
        if error > 10000 or k == 1000:
            raise(RuntimeError('Итерации не сходятся'))
    # после выполнения цикла возвращаем список решений
    return X_res

def simple(A, b, eps, max_iterations=1000):
    """
    Решение СЛАУ методом простых итераций
    
    Параметры:
    A - матрица коэффициентов
    b - вектор правых частей
    eps - точность
    max_iterations - максимальное число итераций
    
    Возвращает:
    X_res - список всех приближений решения
    """
    n = len(A)
    
    # Преобразуем систему к виду x = αx + β
    alpha = [[0.0] * n for _ in range(n)]
    beta = [0.0] * n
    
    for i in range(n):
        beta[i] = b[i] / A[i][i]
        for j in range(n):
            if i != j:
                alpha[i][j] = -A[i][j] / A[i][i]
    
    # Проверяем условие сходимости (достаточное)
    norm_alpha = max(sum(abs(alpha[i][j]) for j in range(n)) for i in range(n))
    if norm_alpha >= 1:
        print(f"Предупреждение: норма матрицы α = {norm_alpha} >= 1, сходимость не гарантирована")
    
    # Задаем начальное приближение
    start = beta[:]  # x^(0) = β
    
    # решения x по всем шагам
    X_res = []
    X_res.append(start)
    
    # номер итерации
    k = 0
    error = float('inf')
    
    # совершаем итерации пока не достигнем желаемой точности
    while error > eps and k < max_iterations:
        k += 1
        
        # получаем значения решений с последней итерации
        x_prev = X_res[-1][:]
        
        # вычисляем новое приближение: x^(k+1) = α * x^(k) + β
        x_new = [0.0] * n
        for i in range(n):
            sum_ax = 0.0
            for j in range(n):
                sum_ax += alpha[i][j] * x_prev[j]
            x_new[i] = sum_ax + beta[i]
        
        # сохраняем новое приближение
        X_res.append(x_new)
        
        # вычисляем ошибку (норма разности приближений)
        error = max(abs(x_new[i] - x_prev[i]) for i in range(n))
        
        # проверка на расходимость
        if error > 1e10 and k > 10:
            raise RuntimeError('Итерации расходятся')
    
    if k == max_iterations:
        print(f"Достигнуто максимальное число итераций {max_iterations}")
    
    return X_res
# eps - точность, которую хотим получить
def jacobi(A, b, eps):
    # первичные значения, с которых начинаем вычисления
    start = [0] * len(A)

    # решения x по всем шагам
    X_res = []
    # пихаем первичные значения (чтобы начать вычисления с них)
    X_res.append(start)
    # задаем максимальную ошибку
    error = float('inf')
    # решение текущего шага
    xn = [0] * len(A)
    # номер итерации
    k = 0
    # совершаем итерации пока не достигнем желаемой точности
    while error > eps:
        # инкрементируем итерацию (нулевая итерация уже сделана, это первичные значения)
        k += 1
        # получаем значения решений с последней итерации
        # используем slice, т.к. прямое присвоение копирует
        # не значение массива, а ссылку на него
        # тогда как slice генерирует новый объект
        x = X_res[-1][:]
        # вычисляем значения текущей итерации по формуле
        for i in range(len(A)):
            xn[i] = (1 / float(A[i][i])) * (b[i] - (fullBracket(A, x, i) - A[i][i] * x[i]))
        # вставляем текущий шаг в список шагов
        X_res.append(xn[:])
        # вычисляем текущую ошибку
        error = errorCalc(x, xn)
        # если никак не можем подобраться к решению, ошибка скачет и прошло много итераций
        if error > 10000 or k == 1000:
            raise(RuntimeError('Итерации не сходятся'))
    # после выполнения цикла возвращаем список решений
    return X_res
def apriori_estimate(A, b, eps):
    """
    Априорная оценка числа итераций для метода простых итераций
    
    Параметры:
    A - матрица коэффициентов
    b - вектор правых частей
    eps - требуемая точность
    
    Возвращает:
    k_min - минимальное число итераций для достижения точности eps
    norm_alpha - норма матрицы α
    """
    n = len(A)
    
    # Преобразуем систему к виду x = αx + β
    alpha = [[0.0] * n for _ in range(n)]
    beta = [0.0] * n
    
    for i in range(n):
        beta[i] = b[i] / A[i][i]
        for j in range(n):
            if i != j:
                alpha[i][j] = -A[i][j] / A[i][i]
    
    # Вычисляем норму матрицы α (первая норма - максимум сумм по строкам)
    norm_alpha = max(sum(abs(alpha[i][j]) for j in range(n)) for i in range(n))
    
    # Вычисляем норму вектора β (используем максимальную норму)
    norm_beta = max(abs(beta_i) for beta_i in beta)
    
    # Проверяем условие сходимости
    if norm_alpha >= 1:
        print(f"Внимание: норма матрицы α = {norm_alpha} >= 1, сходимость не гарантирована")
        return -1, norm_alpha
    
    # Вычисляем минимальное число итераций по формуле (1.19)
    # k + 1 >= (lg(ε) + lg(1 - ||α||) - lg(||β||)) / lg(||α||)
    
    numerator = (math.log10(eps) + math.log10(1 - norm_alpha) - 
                math.log10(norm_beta))
    denominator = math.log10(norm_alpha)
    
    k_plus_1 = numerator / denominator
    
    # Округляем вверх до ближайшего целого
    k_min = math.ceil(k_plus_1) - 1
    
    # Если получилось отрицательное значение, берем хотя бы 1 итерацию
    k_min = max(k_min, 1)
    
    return k_min

A = [[5, 1, 1], [2, 6, -1], [2, 3, 10]]
n = 3
b = [11, 13, 18]
def parameterized_b(b, n):
    new_b = [b[0] + 0.8 * n, b[1] + 0.9 * n, b[2] - 0.5 * n]
    return new_b
b = parameterized_b(b, n)
print("="* 30)
print(" ЗАДАНИЕ 1")
if(isDiagDominant(A)):
    # accuracy = 0.001
    # solution = jacobi(A, b, accuracy)
    # print("Решено за " + str(len(solution) - 1) + " итераций с точностью " + str(accuracy) + " методом Якоби")
    # print("Решение: ")
    # print(solution[-1])

    accuracy = 0.001
    solution = zeidel(A, b, accuracy)
    print("Решено за " + str(len(solution) - 1) + " итераций с точностью " + str(accuracy) + " методом Зейделя")
    print("Решение: ")
    print(solution[-1])

    accuracy = 0.001
    min_k = apriori_estimate(A, b, accuracy)
    solution = simple(A, b, accuracy)
    print("Априорная оценка k: " + str(min_k))
    print("Решено за " + str(len(solution) - 1) + " итераций с точностью " + str(accuracy) + " методом простых итераций")
    print("Решение: ")
    print(solution[-1])
else:
    print("Метод Якоби может не сходиться, матрица не имеет строчного диагонального преобладания")

A = [[12, -3, 2, -1], 
     [-1, 6, -1, 1], 
     [3, 2, -8, 2],
     [2, -1, -1, 5]]
b = [8, 12, -9, 17]
print("="* 30)
print(" ЗАДАНИЕ 2 a)")
if(isDiagDominant(A)):
    # accuracy = 0.001
    # solution = jacobi(A, b, accuracy)
    # print("Решено за " + str(len(solution) - 1) + " итераций с точностью " + str(accuracy) + " методом Якоби")
    # print("Решение: ")
    # print(solution[-1])

    accuracy = 0.001
    solution = zeidel(A, b, accuracy)
    print("Решено за " + str(len(solution) - 1) + " итераций с точностью " + str(accuracy) + " методом Зейделя")
    print("Решение: ")
    print(solution[-1])

    accuracy = 0.001
    min_k = apriori_estimate(A, b, accuracy)
    solution = simple(A, b, accuracy)
    print("Априорная оценка k: " + str(min_k))
    print("Решено за " + str(len(solution) - 1) + " итераций с точностью " + str(accuracy) + " методом простых итераций")
    print("Решение: ")
    print(solution[-1])
else:
    print("Метод Якоби может не сходиться, матрица не имеет строчного диагонального преобладания")

A = [[3, 1, -1, 1], 
     [1, -4, 1, -1], 
     [-1, 1, 4, 1],
     [1, 2, 1, -5]]
m = 4
def parameterized_b(m):
    b = [3 * m, m - 6, 15 - m, m + 2]
    return b
b = parameterized_b(m)
print("="* 30)
print(" ЗАДАНИЕ 2 b)")
if(isDiagDominant(A)):
    # accuracy = 0.001
    # solution = jacobi(A, b, accuracy)
    # print("Решено за " + str(len(solution) - 1) + " итераций с точностью " + str(accuracy) + " методом Якоби")
    # print("Решение: ")
    # print(solution[-1])

    accuracy = 0.001
    solution = zeidel(A, b, accuracy)
    print("Решено за " + str(len(solution) - 1) + " итераций с точностью " + str(accuracy) + " методом Зейделя")
    print("Решение: ")
    print(solution[-1])

    accuracy = 0.001
    min_k = apriori_estimate(A, b, accuracy)
    solution = simple(A, b, accuracy)
    print("Априорная оценка k: " + str(min_k))
    print("Решено за " + str(len(solution) - 1) + " итераций с точностью " + str(accuracy) + " методом простых итераций")
    print("Решение: ")
    print(solution[-1])
else:
    print("Метод Якоби может не сходиться, матрица не имеет строчного диагонального преобладания")