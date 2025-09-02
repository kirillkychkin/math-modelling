import matplotlib.pyplot as plt
import numpy as np


def cos(t):
    return np.cos(2*np.pi*t)
def sin(t):
    return np.sin(2*np.pi*t)

def parabola(t):
    return t**2

def ln(t):
    return np.log(t)

t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.3)
t3 = np.arange(-25.0, 25.0, 0.5)

plt.figure()
plt.subplot(221, title = "cos func", xlabel = "x", ylabel = "cos(x)")
plt.plot(t1, cos(t1), color='tab:blue', marker='o')
plt.plot(t2, cos(t2), color='tab:green', marker='o')
plt.legend(['from 0.0 to 5.0 with step 0.1', 'from 0.0 to 5.0 with step 0.3'])

plt.subplot(222, title = "sin func", xlabel = "x", ylabel = "sin(x)")
plt.plot(t1, sin(t1), color='tab:orange', linestyle='--')

plt.subplot(223, title = "parabola func", xlabel = "x", ylabel = "x^2")
plt.plot(t3, parabola(t3), color='tab:purple', linestyle='dotted')

plt.subplot(224, title = "ln func", xlabel = "x", ylabel = "ln(x)")
plt.plot(t1, parabola(t1), color='tab:red', linestyle='dashed')

plt.show()