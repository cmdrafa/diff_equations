from lagrange_poly import *
from least_squares_funcs import *
import sympy as sym

x = sym.Symbol('x')
#f = sym.sin(2*sym.pi*x)
f = sym.Abs(1 - 2*x)
N = 15
Omega = [0, 1]
psi, points = Lagrange_polynomials(x, N, Omega, point_distribution='Chebyshev')
#u, c = least_squares(f, psi, Omega)
u, c = interpolation(f, psi, points)
comparison_plot(f, u, Omega)