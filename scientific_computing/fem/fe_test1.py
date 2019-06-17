from fe_approx import *

phi = basis(d=1, symbolic=True)
print(phi)
print()
print(element_matrix(phi, Omega_e=[0.1, 0.2], symbolic=True))
print()
print(element_matrix(phi, Omega_e=[0.1, 0.2], symbolic=False))