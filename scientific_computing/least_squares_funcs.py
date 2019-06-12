import sympy as sym
import numpy as np
import matplotlib.pyplot as plt


def least_squares(f, psi, Omega, symbolic=True):
    N = len(psi) - 1
    A = sym.zeros(N+1, N+1)
    b = sym.zeros(N+1, 1)
    x = sym.Symbol('x')
    for i in range(N+1):
        for j in range(i, N+1):
            integrand = psi[i]*psi[j]
            if symbolic:
                I = sym.integrate(integrand, (x, Omega[0], Omega[1]))
            if not symbolic or isinstance(I, sym.Integral):
                integrand = sym.lambdify([x], integrand)
                I = sym.mpmath.quad(integrand, [Omega[0], Omega[1]])
            A[i, j] = A[j, i] = I

        integrand = psi[i]*f
        if symbolic:
            I = sym.integrate(integrand, (x, Omega[0], Omega[1]))
        if not symbolic or isinstance(I, sym.Integral):
            I = sym.mpmath.quad(integrand, [Omega[0], Omega[1]])
        b[i, 0] = I
    c = A.LUsolve(b)
    c = [sym.simplify(c[i, 0]) for i in range(c.shape[0])]
    u = sum(c[i]*psi[i] for i in range(len(psi)))
    return u, c


def comparison_plot(f, u, Omega, filename='tmp.pdf'):
    x = sym.Symbol('x')
    f = sym.lambdify([x], f, modules='numpy')
    u = sym.lambdify([x], u, modules='numpy')
    resolution = 401
    xcoor = np.linspace(Omega[0], Omega[1], resolution)
    exact = f(xcoor)
    approx = u(xcoor)
    plt.figure()
    plt.plot(xcoor, approx, '--')
    plt.plot(xcoor, exact, '-')
    plt.legend(['approx', 'exact'])
    plt.savefig(filename)
    plt.show()


def regression(f, psi, points):
    N = len(psi) - 1
    m = len(points) - 1
    B = np.zeros((N+1, N+1))
    d = np.zeros(N+1)
    x = sym.Symbol('x')
    psi_sym = psi
    psi = [sym.lambdify([x], psi[i]) for i in range(N+1)]
    f = sym.lambdify([x], f)
    for i in range(N+1):
        for j in range(N+1):
            B[i, j] = 0
            for k in range(m+1):
                B[i, j] += psi[i](points[k])*psi[j](points[k])
        d[i] = 0
        for k in range(m+1):
            d[i] += psi[i](points[k])*f(points[k])
    c = np.linalg.solve(B, d)
    u = sum(c[i]*psi_sym[i] for i in range(N+1))
    return u, c
