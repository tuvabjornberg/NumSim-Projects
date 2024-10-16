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
initial = [N - infected, infected, recovered]

t0 = 0
t1 = 120
t_span = [t0, t1]

exposed = 0
alpha = 0.5

dead = 0
my = 0.01

# ---------------- ASSIGNMENT 4 ------ VACCINATIONSMODELL ---------------------
vaccinated = 0
vax_rate = 3

initial_sum = exposed + infected + recovered + dead + vaccinated
initial = [N - initial_sum, exposed, infected, recovered, dead, vaccinated]


# ODE solver, determenistic
def ODE_SEIRDV(t, y):
    s, e, i, r, d, v = y

    s_p = -(beta) * (i / N) * s - vax_rate
    e_p = beta * (i / N) * s - alpha * e
    i_p = alpha * e - (gamma * i) - my * i
    r_p = gamma * i
    d_p = my * i
    v_p = vax_rate

    return np.array([s_p, e_p, i_p, r_p, d_p, v_p])


t_eval = np.arange(t_span[0], t_span[1], 0.1)

sol = solve_ivp(ODE_SEIRDV, t_span, initial, t_eval=t_eval)

plt.plot(sol.t, sol.y[0], label="Susceptible")
plt.plot(sol.t, sol.y[1], label="Exposed")
plt.plot(sol.t, sol.y[2], label="Infected")
plt.plot(sol.t, sol.y[3], label="Recovered")
plt.plot(sol.t, sol.y[4], label="DEATH")
plt.plot(sol.t, sol.y[5], label="Vaccinated")
plt.xlabel("time, t")
plt.ylabel("number of people")
plt.title("Uppgift 4, Vaccination-model solve_ivp")
plt.legend()
plt.show()

# Gillespie, stochastic
initial = (N - initial_sum, exposed, infected, recovered, dead, vaccinated)
coeff = (beta, gamma, alpha, my, vax_rate)


def stochEpedemic():
    M = np.array(
        [
            [-1, 1, 0, 0, 0, 0],
            [0, -1, 1, 0, 0, 0],
            [0, 0, -1, 1, 0, 0],
            [0, 0, -1, 0, 1, 0],
            [-1, 0, 0, 0, 0, 1],
        ]
    )
    return M


def propEpedemic(X, coeff):
    beta = coeff[0]
    gamma = coeff[1]
    alpha = coeff[2]
    my = coeff[3]
    vax_rate = coeff[4]

    s = X[0]
    e = X[1]
    i = X[2]
    r = X[3]
    d = X[4]
    v = X[5]

    w = np.array([beta * (i / N) * s, alpha * e, gamma * i, my * i, vax_rate])
    return w


for i in range(5):
    t, X = gill.SSA(propEpedemic, stochEpedemic, initial, t_span, coeff)

    plt.plot(t, X[:, 0], "b")
    plt.plot(t, X[:, 1], "g")
    plt.plot(t, X[:, 2], "r")
    plt.plot(t, X[:, 3], "m")
    plt.plot(t, X[:, 4], "k")
    plt.plot(t, X[:, 5], "y")

plt.plot(t, X[:, 0], "b", label="Susceptible")
plt.plot(t, X[:, 1], "g", label="Exposed")
plt.plot(t, X[:, 2], "r", label="Infected")
plt.plot(t, X[:, 3], "m", label="Recovered")
plt.plot(t, X[:, 4], "k", label="DEATH")
plt.plot(t, X[:, 5], "y", label="Vaccinated")

plt.xlabel("time, t")
plt.ylabel("number of people")
plt.title("Uppgift 4, Vaccination-model stochastic")
plt.legend()
plt.show()
