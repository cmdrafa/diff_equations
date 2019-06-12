import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate


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


def least_squares_orth(f, psi, Omega, symbolic='true'):
    N = len(psi) - 1
    A = [0]*(N+1)
    b = [0]*(N+1)
    x = sym.Symbol('x')
    for i in range(N+1):
        # Diagonal Matrix term
        A[i] = sym.integrate(psi[i]**2, (x, Omega[0], Omega[1]))

        # Right rand side term
        integrand = psi[i]*f
        if symbolic:
            I = sym.integrate(integrand,  (x, Omega[0], Omega[1]))
        if not symbolic or isinstance(I, sym.Integral):
            print('Numerical integration of', integrand)
            integrand = sym.lambdify([x], integrand)
            I = sym.mpmath.quad(integrand, [Omega[0], Omega[1]])
        b[i] = I
    c = [b[i]/A[i] for i in range(len(b))]
    u = 0
    u = sum(c[i]*psi[i] for i in range(len(psi)))
    return u, c


def least_squares_numerical(f, psi, N, x,
                            integration_method='scipy',
                            orthogonal_basis='off'):

    A = np.zeros((N+1, N+1))
    b = np.zeros(N+1)
    Omega = [x[0], x[-1]]
    dx = x[1] - x[0]

    for i in range(N+1):
        j_limit = i + 1 if orthogonal_basis else N+1
        for j in range(i, j_limit):
            if integration_method == 'scipy':
                A_ij = scipy.integrate.quad(lambda x: psi(x, i)*psi(x, j),
                                            Omega[0], Omega[1], epsabs=1E-9, epsrel=1E-9)[0]
            elif integration_method == 'sympy':
                A_ij = sym.mpmath.quad(
                    lambda x: psi(x, i)*psi(x, j), [Omega[0], Omega[1]])
            else:
                values = psi(x, i)*psi(x, j)
                A_ij = trapezoidal(values, dx)
            A[i, j] = A[j, i] = A_ij

        if integration_method == 'scipy':
            b_i = scipy.integrate.quad(
                lambda x: f(x)*psi(x, i), Omega[0], Omega[1],
                epsabs=1E-9, epsrel=1E-9)[0]
        else:
            values = f(x) * psi(x, i)
            b_i = trapezoidal(values, dx)
        b[i] = b_i

    c = b/np.diag(A) if orthogonal_basis else np.linalg.solve(A, b)
    u = sum(c[i]*psi(x, i) for i in range(N+1))
    return u, c


def trapezoidal(values, dx):
    # Integration by trapezoidal rule(mesh size dx)
    return dx*(np.sum(values) - 0.5*values[0] - 0.5*values[-1])


def comparison_plot(f, u, Omega, filename='tmp.pdf'):
    x = sym.Symbol('x')
    f = sym.lambdify([x], f, modules='numpy')
    u = sym.lambdify([x], u, modules='numpy')
    resolution = 501
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


def interpolation(f, psi, points):
    N = len(psi) - 1
    A = sym.zeros(N+1, N+1)
    b = sym.zeros(N+1, 1)
    psi_sym = psi # save symbolic expression
    # Turn psi and f into Python functions
    x = sym.Symbol('x')
    psi = [sym.lambdify([x], psi[i]) for i in range(N+1)]
    f = sym.lambdify([x], f)
    for i in range(N+1):
        for j in range(N+1):
            A[i,j] = psi[j](points[i])
        b[i,0] = f(points[i])
    c = A.LUsolve(b)
    # c is a sympy Matrix object, turn to list
    c = [sym.simplify(c[i]) for i in range(c.shape[0])]
    u = sym.simplify(sum(c[i]*psi_sym[i] for i in range(N+1)))
    return u, c