from numpy import *


def solver_linear_damping(I, V, m, b, s, F, t):
    N = t.size - 1
    dt = t[1] - t[0]
    u = zeros(N+1)
    u[0] = I
    u[1] = u[0] + dt*V + dt**2/(2*m) * (-b*V - s(u[0]) + F[0])

    for n in range(1, N):
        u[n+1] = 1./(m + b*dt/2)*(2*m*u[n] + \
            (b*dt/2 - m)*u[n-1] + dt**2*(F[n] - s(u[n])))
    return u
