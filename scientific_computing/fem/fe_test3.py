from fe_approx import *
import sympy as sym
import numpy as np

d=1; N_e = 8; Omega = [0,1]
phi = basis(d, symbolic=True)
x = sym.Symbol('x')
f = x*(x-1)
nodes, elements = mesh_uniform(N_e, d, Omega, symbolic=True)
print("Nodes: ")
print(nodes)
print("Elements: ")
print(elements)
A, b = assemble(nodes, elements, phi, f, symbolic=True)
print(A)