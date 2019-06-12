import sympy as sp
x, a = sp.symbols('x a')
Q = a*x**2 - 1 
dQdx = sp.diff(Q, x)
print(dQdx)
Q2 = sp.integrate(dQdx, x)
print(Q2)
Q2 = sp.integrate(Q, (x, 0, a))
print(Q2)
roots = sp.solve(Q, x)
print(roots)

Q = sp.lambdify([x, a], Q)
result = Q(2,3)
print(result)

