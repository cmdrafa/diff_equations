from least_squares_funcs import *

x = sym.Symbol('x')
f = 10*(x-1)**2-1
u, c = least_squares(f=f, psi = [1, x, x**2], Omega=[1,2])
print(u)

print(sym.expand(f))