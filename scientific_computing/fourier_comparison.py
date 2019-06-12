import sympy as sym
from least_squares_funcs import *
N = 17
x = sym.Symbol('x')
psi = [sym.sin(sym.pi*(i+1)*x) for i in range(N+1)]
f = 10*(x-1)**2 - 1
Omega = [0, 1]
u, c = least_squares(f, psi, Omega)
print(c)
comparison_plot(f, u, Omega)
