import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Параметры системы
alpha1 = 1.2    # скорость увеличения популяции жертвы
alpha2 = 0.1    # коэффициент хищничества
alpha3 = 0.8    # смертность хищника
alpha4 = 0.075  # коэффициент роста хищника при поедании жертвы

# Начальные условия
v0 = 15  # начальное количество жертв
w0 = 10  # начальное количество хищников

# Временной интервал
T = 100
t_span = (0, T)

# Система уравнений
def predator_prey(t, y, alpha1, alpha2, alpha3, alpha4):
    v, w = y
    dvdt = (alpha1 - alpha2 * w) * v
    dwdt = (-alpha3 + alpha4 * v) * w
    return [dvdt, dwdt]

# Метод Рунге-Кутта 4-го порядка (вручную)
def runge_kutta_4th_order(f, t_span, y0, n_steps, *args):
    t0, t_end = t_span
    h = (t_end - t0) / n_steps
    t = np.linspace(t0, t_end, n_steps + 1)
    y = np.zeros((n_steps + 1, len(y0)))
    y[0] = y0
    
    for i in range(n_steps):
        k1 = np.array(f(t[i], y[i], *args))
        k2 = np.array(f(t[i] + h/2, y[i] + h/2 * k1, *args))
        k3 = np.array(f(t[i] + h/2, y[i] + h/2 * k2, *args))
        k4 = np.array(f(t[i] + h, y[i] + h * k3, *args))
        
        y[i+1] = y[i] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    return t, y

# Решение методом Рунге-Кутта 4-го порядка
n_steps = 50
t_rk, y_rk = runge_kutta_4th_order(predator_prey, t_span, [v0, w0], n_steps, 
                                   alpha1, alpha2, alpha3, alpha4)
v_rk, w_rk = y_rk[:, 0], y_rk[:, 1]

# Решение с помощью solve_ivp
sol = solve_ivp(predator_prey, t_span, [v0, w0], args=(alpha1, alpha2, alpha3, alpha4),
                method='RK45', dense_output=True, max_step=0.1)
t_ivp = np.linspace(0, T, 50)
y_ivp = sol.sol(t_ivp)
v_ivp, w_ivp = y_ivp[0], y_ivp[1]

# Сравнение решений
print(f"Метод Рунге-Кутта 4-го порядка:")
print(f"  v(T) = {v_rk[-1]:.6f}, w(T) = {w_rk[-1]:.6f}")
print(f"Метод solve_ivp (RK45):")
print(f"  v(T) = {v_ivp[-1]:.6f}, w(T) = {w_ivp[-1]:.6f}")

# Визуализация сравнения
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Графики численности популяций (Рунге-Кутта)
axes[0].plot(t_rk, v_rk, 'b-', label='Жертвы (v)')
axes[0].plot(t_rk, w_rk, 'r-', label='Хищники (w)')
axes[0].set_xlabel('Время')
axes[0].set_ylabel('Численность')
axes[0].set_title('Метод Рунге-Кутта 4-го порядка')
axes[0].legend()
axes[0].grid(True)

# Графики численности популяций (solve_ivp)
axes[1].plot(t_ivp, v_ivp, 'b-', label='Жертвы (v)')
axes[1].plot(t_ivp, w_ivp, 'r-', label='Хищники (w)')
axes[1].set_xlabel('Время')
axes[1].set_ylabel('Численность')
axes[1].set_title('Метод solve_ivp (RK45)')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

def simulate_predator_prey(params, v0, w0, T=100):
    """Функция для симуляции системы с заданными параметрами"""
    alpha1, alpha2, alpha3, alpha4 = params
    
    def system(t, y):
        v, w = y
        dvdt = (alpha1 - alpha2 * w) * v
        dwdt = (-alpha3 + alpha4 * v) * w
        return [dvdt, dwdt]
    
    t_eval = np.linspace(0, T, 1000)
    sol = solve_ivp(system, [0, T], [v0, w0], t_eval=t_eval, method='RK45')
    
    return sol.t, sol.y[0], sol.y[1]

# Базовые параметры
base_params = [alpha1, alpha2, alpha3, alpha4]

# Исследование влияния начальных условий
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Влияние начальных условий на динамику системы', fontsize=16)

# Варианты начальных условий
initial_conditions = [
    (15, 10, 'Базовый случай'),
    (5, 5, 'Маленькие популяции'),
    (30, 5, 'Много жертв, мало хищников'),
    (10, 20, 'Мало жертв, много хищников')
]

for idx, (v0, w0, title) in enumerate(initial_conditions):
    t, v, w = simulate_predator_prey(base_params, v0, w0)
    
    ax = axes[idx // 2, idx % 2]
    ax.plot(t, v, 'b-', label='Жертвы')
    ax.plot(t, w, 'r-', label='Хищники')
    ax.set_xlabel('Время')
    ax.set_ylabel('Численность')
    ax.set_title(f'{title}\nv0={v0}, w0={w0}')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()

# Исследование влияния коэффициентов
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Влияние коэффициентов на динамику системы', fontsize=16)

# Варианты параметров
param_variations = [
    ([1.5, alpha2, alpha3, alpha4], 'Увеличение α1 (рождаемость жертв)'),
    ([alpha1, 0.05, alpha3, alpha4], 'Уменьшение α2 (хищничество)'),
    ([alpha1, alpha2, 0.5, alpha4], 'Уменьшение α3 (смертность хищников)'),
    ([alpha1, alpha2, alpha3, 0.1], 'Увеличение α4 (рост хищников)')
]

for idx, (params, title) in enumerate(param_variations):
    t, v, w = simulate_predator_prey(params, v0, w0)
    
    ax = axes[idx // 2, idx % 2]
    ax.plot(t, v, 'b-', label='Жертвы')
    ax.plot(t, w, 'r-', label='Хищники')
    ax.set_xlabel('Время')
    ax.set_ylabel('Численность')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()

# Анализ устойчивости системы
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Анализ устойчивости системы при различных параметрах', fontsize=16)

# Различные сценарии
scenarios = [
    ([0.8, 0.1, 0.8, 0.075], 'Низкая рождаемость жертв'),
    ([1.2, 0.2, 0.8, 0.075], 'Высокое хищничество'),
    ([1.2, 0.1, 1.0, 0.075], 'Высокая смертность хищников'),
    ([1.2, 0.1, 0.8, 0.05], 'Низкий рост хищников'),
    ([1.5, 0.05, 0.5, 0.1], 'Благоприятные условия'),
    ([0.5, 0.2, 1.0, 0.05], 'Неблагоприятные условия')
]

for idx, (params, title) in enumerate(scenarios):
    t, v, w = simulate_predator_prey(params, v0, w0)
    
    ax = axes[idx // 3, idx % 3]
    ax.plot(t, v, 'b-', label='Жертвы')
    ax.plot(t, w, 'r-', label='Хищники')
    ax.set_xlabel('Время')
    ax.set_ylabel('Численность')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()