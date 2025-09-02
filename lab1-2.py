import matplotlib.pyplot as plt
import numpy as np

def f(x, y):
    return x**2 - 2*x*y + 3*y**2

x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)

Z = f(X, Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(X, Y)')
ax.set_title('Surface plot of f(x, y) = x^2 - 2*x*y + 3*y^2')

plt.show()