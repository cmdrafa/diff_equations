from hd_funcs import least_squares_2d, comparison_plot
import sympy as sym


def taylor(x, y, Nx, Ny):
    return [x**i*y**j for i in range(Nx+1) for j in range(Ny+1)]


def sines(x, y, Nx, Ny):
    return [sym.sin(sym.pi*(i+1)*x)*sym.sin(sym.pi*(j+1)*y)
            for i in range(Nx+1) for j in range(Ny + 1)]

x, y = sym.symbols('x y')
f = (1+x**2)*(1+2*y**2)
psi = taylor(x, y, 2, 2)
print(psi)
Omega = [[0, 2], [0, 2]]
u = least_squares_2d(f, psi, Omega, symbolic=False)
comparison_plot(f, u, Omega)
