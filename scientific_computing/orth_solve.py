import sympy as sym
from least_squares_funcs import least_squares, comparison_plot, least_squares_orth

N = 3
x = sym.Symbol('x')
psi = [sym.sin(sym.pi*(i+1)*x) for i in range(N+1)]
f0 = 9; f1 = -1
f = 10*(x-1)**2 - 1
Omega = [0, 1]
B = f0*(1-x) + x*f1
u_sum, c = least_squares(f-B, psi, Omega)
u = B + u_sum
comparison_plot(f, u, Omega)