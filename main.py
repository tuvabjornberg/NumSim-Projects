import numpy as np
import scipy
from scipy.integrate import solve_ivp

g = 9.82
c = 0.05
k = 700


def mass(t):
    if t <= 10:
        return 8 - 0.4 * t
    else:
        return 4


def mass_der(t):
    if t <= 10:
        return -0.4
    else:
        return 0


def engine_dir(t, height):
    if height < 20:
        return np.pi / 2
    else:
        return t  # TODO: Stub


def fuel_velocity(t, height):
    direction = engine_dir(t, height)
    x = k * np.cos(direction)
    y = k * np.sin(direction)
    return (x, y)


def external_forces(t, v):
    m = mass(t)
    gravity = (0, m * (-g))  # Gravity vector: x, y direction
    air_res = c * np.linalg.norm(v) * v  # Air resistance: c||v(t)||v(t)
    return gravity + air_res


def ode_rhs(t, v, height):
    # dv(t)/dt = g - c||v(t)||v(t) + m'(t)u(t)/m(t)
    # --> dv(t)/dt = (external_forces() + mass_der()fuel_velocity())/mass()

    m = mass(t)
    md = mass_der(t)
    fv = np.array(fuel_velocity(t, height))
    ext_f = np.array(external_forces(t, v))

    dvdt = (ext_f + md * fv) / m
    # return dvdt
