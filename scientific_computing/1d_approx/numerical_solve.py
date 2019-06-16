import numpy as np
from least_squares_funcs import least_squares_numerical, comparison_plot

def psi(x, i):
    return np.sin((i+1)*x)

x = np.linspace(0, 2*np.pi, 501)
f = 10*(x-1)**2 - 1
Omega = [0, 1]
N = 20
u, c = least_squares_numerical(lambda x: np.tanh(x-np.pi), psi, N, x, orthogonal_basis=True)
comparison_plot(f, u, Omega)

print(u)