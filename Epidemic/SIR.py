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

initial_sum = infected + recovered
initial = [N - initial_sum, infected, recovered]

t0 = 0
t1 = 120
t_span = [t0, t1]


# ---------------- ASSIGNEMNT 1 ------ SIR MODEL ---------------------
# ODE solver, determenistic
def ODE_SIR(t, y):
    s, i, r = y

    s_p = -(beta) * (i / N) * s
    i_p = beta * (i / N) * s - (gamma * i)
    r_p = gamma * i

    return np.array([s_p, i_p, r_p])


t_eval = np.arange(t_span[0], t_span[1], 0.1)

sol = solve_ivp(ODE_SIR, t_span, initial, t_eval=t_eval)

plt.plot(sol.t, sol.y[0], label="Susceptible")
plt.plot(sol.t, sol.y[1], label="Infected")
plt.plot(sol.t, sol.y[2], label="Recovered")
plt.xlabel("time, t")
plt.ylabel("number of people")
plt.title("Uppgift 1, SIR-model solve_ivp")
plt.legend()
plt.show()

# Gillespie, stochastic
initial = (N - initial_sum, infected, recovered)
coeff = (beta, gamma)


def stochEpedemic():
    M = np.array([[-1, 1, 0], [0, -1, 1]])
    return M


def propEpedemic(X, coeff):
    beta = coeff[0]
    gamma = coeff[1]

    s = X[0]
    i = X[1]
    r = X[2]

    w = np.array([beta * (i / N) * s, gamma * i])
    return w


for i in range(5):
    t, X = gill.SSA(propEpedemic, stochEpedemic, initial, t_span, coeff)

    plt.plot(t, X[:, 0], "b")
    plt.plot(t, X[:, 1], "g")
    plt.plot(t, X[:, 2], "r")

plt.plot(t, X[:, 0], "b", label="Susceptible")
plt.plot(t, X[:, 1], "g", label="Infected")
plt.plot(t, X[:, 2], "r", label="Recovered")

plt.xlabel("time, t")
plt.ylabel("number of people")
plt.title("Uppgift 1, SIR-model stochastic")
plt.legend()
plt.show()
