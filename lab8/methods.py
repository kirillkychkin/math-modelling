import numpy as np

# Метод Эйлера
def euler_method(x, y0, n, h, f):
    y = np.zeros(n + 1)
    y[0] = y0
    
    for i in range(n):
        y[i + 1] = y[i] + h * f(x[i], y[i])
    
    return y

# Модифицированный метод Эйлера
def modified_euler_method(x, y0, n, h, f):
    y = np.zeros(n + 1)
    y[0] = y0
    
    for i in range(n):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + h, y[i] + h * k1)
        y[i + 1] = y[i] + h * (k1 + k2) / 2
    
    return y

# Метод Рунге-Кутта 4-го порядка
def runge_kutta_method(x, y0, n, h, f):
    y = np.zeros(n + 1)
    y[0] = y0
    
    for i in range(n):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + h/2, y[i] + h/2 * k1)
        k3 = f(x[i] + h/2, y[i] + h/2 * k2)
        k4 = f(x[i] + h, y[i] + h * k3)
        
        y[i + 1] = y[i] + h * (k1 + 2*k2 + 2*k3 + k4) / 6
    
    return y