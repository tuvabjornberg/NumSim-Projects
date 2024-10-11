import numpy as np
import scipy
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import gillespie as gill

beta = 0.3
gamma = 1 / 7

N = 1000
infected = 5
recovered = 0

t0 = 0
t1 = 120
t_span = [t0, t1]

# From SEIR
exposed = 0
alpha = 0.5

# From SEIRD
dead = 0
my = 0.01

# From SERIDV
vaccinated = 0
vax_rate = 3


# SIERDV
def ODE_SEIRDV(t, y):
    s, e, i, r, d, v = y

    s_p = -(beta) * (i / N) * s - vax_rate
    e_p = beta * (i / N) * s - alpha * e
    i_p = alpha * e - (gamma * i) - my * i
    r_p = gamma * i
    d_p = my * i
    v_p = vax_rate

    return np.array([s_p, e_p, i_p, r_p, d_p, v_p])


initial_SEIRDV = [N - infected, exposed, infected, recovered, dead, vaccinated]
t_eval = np.arange(t_span[0], t_span[1], 0.1)

sol_SEIRDV = solve_ivp(ODE_SEIRDV, t_span, initial_SEIRDV, t_eval=t_eval)

# ---------------- ASSIGNMENT 5 ------ EGEN SMITTSPRIDNINGSMODELL ---------------------

delta = 0.5  # rate av R -> S
epsilon = 0.94  # protection rate
zeta = 0.3  # rate av R -> Im
second_dose_rate = 5
immune = 0
initial = [N - infected, exposed, infected, recovered, dead, vaccinated, immune]


# ODE solver, determenistic
def ODE_5(t, y):
    s, e, i, r, d, v, im = y

    s_p = -(beta) * (i / N) * s - vax_rate + delta * r + (1 - epsilon) * v
    e_p = beta * (i / N) * s - alpha * e
    i_p = alpha * e - (gamma * i) - my * i
    r_p = gamma * i - delta * r - zeta * r
    d_p = my * i
    v_p = vax_rate - (1 - epsilon) * v - min(second_dose_rate, v)
    im_p = min(second_dose_rate, v) + zeta * r

    return np.array([s_p, e_p, i_p, r_p, d_p, v_p, im_p])


t_eval = np.arange(t_span[0], t_span[1], 0.1)

sol = solve_ivp(ODE_5, t_span, initial, t_eval=t_eval)

s, e, i, r, d, v, im = sol.y
total_population = s + e + i + r + d + v + im

# Plot to check if total population remains constant
plt.plot(sol.t, total_population, label="Total Population")
plt.axhline(y=N, color="r", linestyle="--", label="Expected N=1000")

plt.plot(sol.t, sol.y[0], label="Susceptible")
plt.plot(sol.t, sol.y[1], label="Exposed")
plt.plot(sol.t, sol.y[2], label="Infected")
plt.plot(sol.t, sol.y[3], label="Recovered")
plt.plot(sol.t, sol.y[4], label="DEATH")
plt.plot(sol.t, sol.y[5], label="Vaccinated")
plt.plot(sol.t, sol.y[6], label="Immune")

plt.plot(sol_SEIRDV.t, sol_SEIRDV.y[0], label="Susceptible OLD")
plt.plot(sol_SEIRDV.t, sol_SEIRDV.y[1], label="Exposed OLD")
plt.plot(sol_SEIRDV.t, sol_SEIRDV.y[2], label="Infected OLD")
plt.plot(sol_SEIRDV.t, sol_SEIRDV.y[3], label="Recovered OLD")
plt.plot(sol_SEIRDV.t, sol_SEIRDV.y[4], label="DEATH OLD")
plt.plot(sol_SEIRDV.t, sol_SEIRDV.y[5], label="Vaccinated OLD")


plt.xlabel("time, t")
plt.ylabel("number of people")
plt.title("Uppgift 5, Egen Vaccination-model solve_ivp")
plt.legend()
plt.show()


# Gillespie, stochastic
initial = (N - infected, exposed, infected, recovered, dead, vaccinated, immune)
coeff = (beta, gamma, alpha, my, vax_rate, delta, epsilon, second_dose_rate, zeta)


def stochEpedemic():
    M = np.array(
        [
            [-1, 1, 0, 0, 0, 0, 0],
            [0, -1, 1, 0, 0, 0, 0],
            [0, 0, -1, 1, 0, 0, 0],
            [0, 0, -1, 0, 1, 0, 0],
            [-1, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, -1, 0, 0, 0],
            [1, 0, 0, 0, 0, -1, 0],
            [0, 0, 0, 0, 0, -1, 1],
            [0, 0, 0, -1, 0, 0, 1],
        ]
    )
    return M


def propEpedemic(X, coeff):
    beta = coeff[0]
    gamma = coeff[1]
    alpha = coeff[2]
    my = coeff[3]
    vax_rate = coeff[4]
    delta = coeff[5]
    epsilon = coeff[6]
    second_dose_rate = coeff[7]
    zeta = coeff[8]

    s = X[0]
    e = X[1]
    i = X[2]
    r = X[3]
    d = X[4]
    v = X[5]
    im = X[6]

    w = np.array(
        [
            beta * (i / N) * s,
            alpha * e,
            gamma * i,
            my * i,
            vax_rate,
            delta * r,
            (1 - epsilon) * v,
            min(second_dose_rate, v),
            zeta * r,
        ]
    )
    return w


for i in range(5):
    t, X = gill.SSA(propEpedemic, stochEpedemic, initial, t_span, coeff)

    plt.plot(t, X[:, 0], "b")
    plt.plot(t, X[:, 1], "g")
    plt.plot(t, X[:, 2], "r")
    plt.plot(t, X[:, 3], "m")
    plt.plot(t, X[:, 4], "k")
    plt.plot(t, X[:, 5], "y")
    plt.plot(t, X[:, 6], "tab:pink")

total_population = X[:, 0] + X[:, 1] + X[:, 2] + X[:, 3] + X[:, 4] + X[:, 5] + X[:, 6]

# Plot to check if total population remains constant
plt.plot(t, total_population, label="Total Population")
plt.axhline(y=N, color="r", linestyle="--", label="Expected N=1000")

plt.plot(t, X[:, 0], "b", label="Susceptible")
plt.plot(t, X[:, 1], "g", label="Exposed")
plt.plot(t, X[:, 2], "r", label="Infected")
plt.plot(t, X[:, 3], "m", label="Recovered")
plt.plot(t, X[:, 4], "k", label="DEATH")
plt.plot(t, X[:, 5], "y", label="Vaccinated")
plt.plot(t, X[:, 6], "tab:pink", label="Immune")

plt.xlabel("time, t")
plt.ylabel("number of people")
plt.title("Uppgift 5, Egen Vaccination-model stochastic")
plt.legend()
plt.show()
