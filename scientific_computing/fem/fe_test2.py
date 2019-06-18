from fe_approx import *
import sympy as sym
import numpy as np

#h, x = sym.symbols('h x')
#nodes = [0, h, 2*h]
nodes = [0, 0.5, 1]
elements = [[0,1], [1,2]]
phi = basis(d=1, symbolic=True)
x = sym.Symbol('x')
f = x*(1-x)
#A, b = assemble(nodes, elements,phi, f, symbolic=True)
A, b = assemble(nodes, elements,phi, f, symbolic=False)
#c = A.LUsolve(b)
c = np.linalg.solve(A, b)
print("C")
print(c)