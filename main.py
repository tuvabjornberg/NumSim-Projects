import numpy as np
import scipy
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

g = 9.82
c = 0.05
k = 700

target_x = 80
target_y = 60
initial_rocket_mass = 8

t_span = (0, 100)  # time intervall
initial_state = [
    0,
    0,
    0,
    0,
    initial_rocket_mass,
]  # [x_pos, y_pos, x_velocity, y_velocity, mass]


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


def engine_dir(t, x, y):
    if y < 20:
        return -np.pi / 2
    else:
        Ground = np.array(60, 0)
        v = np.array([x, y])
        v_len = np.sqrt((60 - x) ** 2 + (80 - y) ** 2)

        angle = np.arccos((v.dot(Ground)) / v_len * 60)
        return angle + 2*np.pi


def fuel_velocity(t, x, y):
    if t > 10:
        return np.array([0, 0])
    else:
        direction = engine_dir(t, x, y)
        vx = k * np.cos(direction)
        vy = k * np.sin(direction)
        return np.array([vx, vy])


def external_forces(t, v):
    m = mass(t)
    gravity_F = (0, m * (-g))  # Gravity vector: x, y direction
    air_res = -c * np.linalg.norm(v) * v  # Air resistance: c||v(t)||v(t)
    return np.array([gravity_F + air_res])


def rocket_velocity(t, v, state):
    # dv(t)/dt = g - c||v(t)||v(t) + m'(t)u(t)/m(t)
    # --> dv(t)/dt = (external_forces() + mass_der()fuel_velocity())/mass()
    x, y, vx, vy, m = state
    v = np.array([vx, vy])

    m = mass(t)
    md = mass_der(t)
    fv = fuel_velocity(t, x, y)
    ext_f = external_forces(t, v)

    dvdt = (ext_f + md * fv) / m
    return dvdt


def func(t, state):
    x, y, vx, vy, m = state

    v = np.array([vx, vy])
    ext_forces = external_forces(t, v)
    velocity = fuel_velocity(t, x, y)

    total_force_x = ext_forces[0] + velocity[0]
    total_force_y = ext_forces[1] + velocity[1]

    acc_x = total_force_x * m
    acc_y = total_force_y * m

    mass_change_rate = mass_der(t)

    state_derivatives = [
        vx,
        vy,
        acc_x,
        acc_y,
        mass_change_rate
    ]

    return state_derivatives  # [x_pos, y_pos, x_velocity, y_velocity, m]


sol = solve_ivp(func, t_span, initial_state)

plt.plot(sol.t, sol.y[0])
plt.title("Rocket steering")
plt.xlabel("Position x ()")
plt.ylabel("Position y ()")
plt.grid()
plt.show()
