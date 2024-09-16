import numpy as np
import scipy
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

g = 9.82
c = 0.05
k = 700

burn_rate = 0.4
initial_rocket_mass = 8
initial_fuel_mass = 4
t_max = initial_fuel_mass / burn_rate

target_x = 80
target_y = 60
target = (target_x, target_y)

t_span = (0, 10)  # time intervall
initial_state = [0, 0, 0, 0]  # [x_pos, y_pos, x_velocity, y_velocity, mass]


def mass(t):
    if t <= t_max:  # 4 / 0.4
        return initial_rocket_mass - burn_rate * t
    else:
        return initial_rocket_mass - initial_fuel_mass


def mass_der(t):
    if t <= t_max:
        return -burn_rate
    else:
        return 0


def engine_dir(t, x, y, target):
    x_t, y_t = target
    if y < 20:
        return -np.pi / 2
    else:
        angle = np.tan((y_t - y) / (x_t - x))
        return angle + np.pi


def fuel_velocity(t, x, y, target, steering_func):
    if t > t_max:
        return np.array([0, 0])
    else:
        direction = steering_func(t, x, y, target)
        vx = k * np.cos(direction)
        vy = k * np.sin(direction)
        return np.array([vx, vy])


def external_forces(t, v):
    m = mass(t)
    gravity_F = m * np.array([0, -g])  # Gravity vector: x, y direction
    air_res = -c * np.linalg.norm(v) * v  # Air resistance: c||v(t)||v(t)
    return gravity_F + air_res


def rocket_ODE(t, y, steering_func):
    x_pos, y_pos, vx, vy = y

    m = mass(t)
    v = np.array([vx, vy])

    ext_forces = external_forces(t, v)
    engine_forces = mass_der(t) * fuel_velocity(t, x_pos, y_pos, target, steering_func)
    total_force = ext_forces + engine_forces

    acc = total_force / m

    state_derivatives = [vx, vy, acc[0], acc[1]]
    return state_derivatives  # [x_pos, y_pos, x_velocity, y_velocity, m]


def RK4(f, tspan, u0, dt, *args):
    t_vec = np.arange(tspan[0], tspan[1] + 1.0e-14, dt)
    dt_vec = dt * np.ones_like(t_vec)
    if t_vec[-1] < tspan[1]:
        t_vec = np.append(t_vec, tspan[1])
        dt_vec = np.append(dt_vec, t_vec[-1] - t_vec[-2])

    u = np.zeros((len(t_vec), len(u0)))
    u[0, :] = u0
    for i in range(len(t_vec) - 1):
        h = dt_vec[i]
        k1 = np.array(f(t_vec[i], u[i, :], *args))
        k2 = np.array(f(t_vec[i] + 0.5 * h, u[i, :] + 0.5 * h * k1, *args))
        k3 = np.array(f(t_vec[i] + 0.5 * h, u[i, :] + 0.5 * h * k2, *args))
        k4 = np.array(f(t_vec[i + 1], u[i, :] + h * k3, *args))
        u[i + 1, :] = u[i, :] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return t_vec, u


tt = np.arange(t_span[0], t_span[1], 0.1)
sol = solve_ivp(rocket_ODE, t_span, initial_state, args=(engine_dir,), t_eval=tt)
t, u = RK4(rocket_ODE, t_span, initial_state, 0.1, engine_dir)

# --------- X-Y TO TIME AXIS ---------
plt.plot(sol.t, sol.y[0], label="x position")
plt.plot(sol.t, sol.y[1], label="y position")
plt.title("Rocket steering")
plt.xlabel("Position x ()")
plt.ylabel("Position y ()")
plt.grid()
plt.legend()
plt.show()

# --------- X-Y POS AXES ---------
plt.plot(sol.y[0], sol.y[1])
plt.title("Rocket steering - xy pos")
plt.xlabel("Position x ()")
plt.ylabel("Position y ()")
plt.xlim([-5, 100])
plt.ylim([-5, 100])
plt.grid()
plt.prism()
plt.show()

# --------- RUNGEKUTTA ---------
plt.plot(u[:, 0], u[:, 1])
plt.title("Rocket steering - xy pos RUNGEKUTTA")
plt.xlabel("Position x ()")
plt.ylabel("Position y ()")
plt.xlim([-5, 100])
plt.ylim([-5, 100])
plt.grid()
plt.show()
