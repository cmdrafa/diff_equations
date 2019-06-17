import sympy as sym


def Lagrange_polynomial(x, i, points):
    p = 1
    for k in range(len(points)):
        if k != i:
            p *= (x - points[k])/(points[i] - points[k])
    return p


def Lagrange_polynomials_01(x, N):
    if isinstance(x, sym.Symbol):
        h = sym.Rational(1, N-1)
    else:
        h = 1.0/(N-1)
    points = [i*h for i in range(N)]
    psi = [Lagrange_polynomial(x, i, points) for i in range(N)]
    return psi, points


def Lagrange_polynomials(x, N, Omega, point_distribution='uniform'):
    if point_distribution == 'uniform':
        if isinstance(x, sym.Symbol):
            h = sym.Rational(Omega[1] - Omega[0], N)
        else:
            h = (Omega[1] - Omega[0])/float(N)
            points = [Omega[0] + i*h for i in range(N+1)]
    elif point_distribution == 'Chebyshev':
        points = Chebyshev_Nodes(Omega[0], Omega[1], N)
        psi = [Lagrange_polynomial(x, i, points) for i in range(N+1)]
        return psi, points


def Chebyshev_Nodes(a, b, N):
    from math import cos, pi
    half = 0.5
    nodes = [half*(a+b) + half*(b-a)*cos(float(2*i+1)/(2*(N+1))*pi)
             for i in range(N+1)]
    return nodes
