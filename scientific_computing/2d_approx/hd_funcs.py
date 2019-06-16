import sympy as sym
from mpmath import *
import numpy as np


def least_squares_2d(f, psi, Omega, symbolic=True, print_latex=False):
    N = len(psi) - 1
    A = sym.zeros(N+1, N+1)
    b = sym.zeros(N+1, 1)
    x, y = sym.symbols('x y')
    for i in range(N+1):
        for j in range(i, N+1):
            print("i: j: ", [i, j])
            integrand = psi[i] * psi[j]
            if symbolic:
                I = sym.integrate(integrand,
                                  (x, Omega[0][0], Omega[0][1]),
                                  (y, Omega[1][0], Omega[1][1]))
            if not symbolic or isinstance(I, sym.Integral):
                integrand = sym.lambdify([x, y], integrand)
                I = quad(integrand, [Omega[0][0], Omega[0][1]], [
                                    Omega[1][0], Omega[1][1]])
            A[i, j] = A[j, i] = I
        integrand = psi[i]*f
        if symbolic:
            I = sym.integrate(
                integrand, (x, Omega[0][0], Omega[0][1]), (y, Omega[1][0], Omega[1][1]))
        if not symbolic or isinstance(I, sym.Integral):
            integrand = sym.lambdify([x, y], integrand)
            I = quad(integrand, [Omega[0][0], Omega[0][1]], [
                                Omega[1][0], Omega[1][1]])
        b[i] = I
    print("A: \n", A, "\nb:\n", b)
    if symbolic:
        c = A.LUsolve(b)
        c = [c[i] for i in range(c.shape[0])]
    else:
        c = lu_solve(A, b)
    print("Coeff:", c)

    u = sum(c[i]*psi[i] for i in range(len(psi)))
    print("Approximation: ", u)
    print("F: ", sym.expand(f))
    return u, c


def comparison_plot(f, u, Omega, plotfile='tmp', title=''):

    x, y = sym.symbols('x y')

    f = sym.lambdify([x, y], f, modules="numpy")
    u = sym.lambdify([x, y], u, modules="numpy")
    # When doing symbolics, Omega can easily contain symbolic expressions,
    # assume .evalf() will work in that case to obtain numerical
    # expressions, which then must be converted to float before calling
    # linspace below
    for r in range(2):
        for s in range(2):
            if not isinstance(Omega[r][s], (int, float)):
                Omega[r][s] = float(Omega[r][s].evalf())

    resolution = 41  # no of points in plot
    xcoor = np.linspace(Omega[0][0], Omega[0][1], resolution)
    ycoor = np.linspace(Omega[1][0], Omega[1][1], resolution)
    xv, yv = np.meshgrid(xcoor, ycoor)
    # Vectorized functions expressions does not work with
    # lambdify'ed functions without the modules="numpy"
    exact = f(xv, yv)
    print(xv, yv)
    approx = u(xv, yv)
    print(type(exact))
    print(type(approx))
    error = exact - approx

    import matplotlib.pyplot as plt
    plt.figure()
    contours = plt.contour(xv, yv, error, 8)     # 8 contour lines
    plt.clabel(contours, inline=1, fontsize=10)  # labels
    if title:
        plt.title(title)
    if plotfile:
        plt.savefig('%s_error_c.pdf' % plotfile)
        plt.savefig('%s_error_c.png' % plotfile)
    fig = plt.figure()
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    # ax = fig.add_subplot(111) #, projection='3d')
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(xv, yv, error, rstride=1, cstride=1,
                           cmap=cm.coolwarm, linewidth=0,
                           antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    if title:
        plt.title(title)
    if plotfile:
        plt.savefig('%s_error_s.pdf' % plotfile)
        plt.savefig('%s_error_s.png' % plotfile)
    plt.show()
