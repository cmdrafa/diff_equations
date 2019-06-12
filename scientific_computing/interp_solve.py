import sympy as sym
from least_squares_funcs import *

x = sym.Symbol('x')
f = 10*(x-1)**2 - 1
psi = [1, x]
Omega = [1, 2]
points = [1 + sym.Rational(1, 3), 1 + sym.Rational(2, 3)]
print(points)
u, c = interpolation(f, psi, points)

comparison_plot(f, u, Omega)
