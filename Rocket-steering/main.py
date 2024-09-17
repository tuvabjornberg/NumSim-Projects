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
neg_target = (-40, 80)
high_target = (20, 90)

t_span = (0, 10 + 1.0e-14)  # time intervall
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


# First strategy, only goos when rocket is to the left and below the target, and in the first quadrant
def engine_dir_tan(t, state, target):
    x_pos, y_pos, vx, vy = state
    x_t, y_t = target
    if y_pos < 20:
        return -np.pi / 2
    else:
        angle = np.tan((y_t - y_pos) / (x_t - x_pos))
        return angle + np.pi


def engine_dir_arctan(t, state, target):
    x_pos, y_pos, vx, vy = state

    x_t, y_t = target
    if y_pos < 20:
        return -np.pi / 2
    else:
        angle = np.arctan2((y_t - y_pos), (x_t - x_pos))
        return angle + np.pi


def engine_dir_corrected(t, state, target):
    x_pos, y_pos, vx, vy = state

    x_t, y_t = target
    dx = x_t - x_pos
    dy = y_t - y_pos

    distance = np.sqrt(dx**2 + dy**2)

    if y_pos < 20:
        return -np.pi / 2
    else:
        angle = np.arctan2(dy, dx)
        if x_pos < x_t:
            corrected_angle = angle / (distance * 0.05)
        else:
            corrected_angle = angle + angle / (distance * 0.05)

        return corrected_angle + np.pi


def engine_dir(t, state, target):
    x_pos, y_pos, vx, vy = state

    x_t, y_t = target
    dx = x_t - x_pos
    dy = y_t - y_pos

    current_dir = np.arctan2(vy, vx)

    if y_pos < 20:
        return -np.pi / 2
    else:
        angle = np.arctan2(dy, dx)
        diff = angle - current_dir
        corrected_angle = angle + diff

        return corrected_angle + np.pi


def fuel_velocity(t, state, target, steering_func):
    x_pos, y_pos, vx, vy = state

    if t > t_max:
        return np.array([0, 0])
    else:
        direction = steering_func(t, state, target)
        vx = k * np.cos(direction)
        vy = k * np.sin(direction)
        return np.array([vx, vy])


def external_forces(t, v):
    m = mass(t)
    gravity_F = m * np.array([0, -g])  # Gravity vector: x, y direction
    air_res = -c * np.linalg.norm(v) * v  # Air resistance: c||v(t)||v(t)
    return gravity_F + air_res


def rocket_ODE(t, y, steering_func, target):
    x_pos, y_pos, vx, vy = y

    m = mass(t)
    v = np.array([vx, vy])

    ext_forces = external_forces(t, v)
    engine_forces = mass_der(t) * fuel_velocity(t, y, target, steering_func)
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
sol_norm = solve_ivp(
    rocket_ODE, t_span, initial_state, args=(engine_dir, target), t_eval=tt
)
t_norm, u_norm = RK4(rocket_ODE, t_span, initial_state, 0.1, engine_dir, target)

sol_neg = solve_ivp(
    rocket_ODE, t_span, initial_state, args=(engine_dir, neg_target), t_eval=tt
)
t_neg, u_neg = RK4(rocket_ODE, t_span, initial_state, 0.1, engine_dir, neg_target)

sol_high = solve_ivp(
    rocket_ODE, t_span, initial_state, args=(engine_dir, high_target), t_eval=tt
)
t_high, u_high = RK4(rocket_ODE, t_span, initial_state, 0.1, engine_dir, high_target)

sol_arctan = solve_ivp(
    rocket_ODE, t_span, initial_state, args=(engine_dir_arctan, target), t_eval=tt
)

# --------- X-Y TO TIME AXIS ---------
# plt.plot(sol.t, sol.y[0], label="x position")
# plt.plot(sol.t, sol.y[1], label="y position")
# plt.title("Rocket steering")
# plt.xlabel("Position x ()")
# plt.ylabel("Position y ()")
# plt.grid()
# plt.legend()
# plt.show()


# --------- X-Y POS AXES ---------
plt.plot(sol_norm.y[0], sol_norm.y[1], label="Rocket normal")
plt.plot(target_x, target_y, "ro", label="Target normal (80, 60)")

plt.plot(sol_neg.y[0], sol_neg.y[1], label="Rocket neg")
plt.plot(neg_target[0], neg_target[1], "go", label="Target neg (-40, 80)")

plt.plot(sol_high.y[0], sol_high.y[1], label="Rocket high")
plt.plot(high_target[0], high_target[1], "yo", label="Target high(-40, 90)")

plt.plot(sol_arctan.y[0], sol_arctan.y[1], label="Rocket old arctan")

plt.title("Rocket steering - xy pos")
plt.xlabel("Position x ()")
plt.ylabel("Position y ()")
# plt.xlim([-5, 100])
# plt.ylim([-5, 100])
plt.grid()
plt.prism()
plt.legend()
plt.show()

# --------- RUNGEKUTTA ---------
plt.plot(u_norm[:, 0], u_norm[:, 1], label="Rocket RK normal")
plt.plot(target_x, target_y, "ro", label="Target normal (80, 60)")

plt.plot(u_neg[:, 0], u_neg[:, 1], label="Rocket RK neg")
plt.plot(neg_target[0], neg_target[1], "go", label="Target neg (-40, 80)")

plt.plot(u_high[:, 0], u_high[:, 1], label="Rocket RK high")
plt.plot(high_target[0], high_target[1], "yo", label="Target (-40, 90)")
plt.title("Rocket steering - xy pos RUNGEKUTTA")
plt.xlabel("Position x ()")
plt.ylabel("Position y ()")
# plt.xlim([-5, 100])
# plt.ylim([-5, 100])
plt.grid()
plt.legend()
plt.show()
