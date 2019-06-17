from fe_approx import *
import sympy as sym

h, x = sym.symbols('h x')
nodes = [0, h, 2*h]
elements = [[0,1], [1,2]]
phi = basis(d=1, symbolic=True)
print("PHI: ")
print(phi)
f = x*(1-x)
A, b = assemble(nodes, elements,phi, f, symbolic=True)
print(A)
print(b)
c = A.LUsolve(b)
print(c)