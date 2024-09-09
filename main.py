import numpy as np 
import scipy

def mass(t):
    if t <= 10:
        return 8 - 0.4 * t
    else:
        return 4

def engine_dir(t):
    return t

def velocity(t):
    k = 700
    direction = engine_dir(t)
    x = k * np.cos(direction)
    y = k * np.sin(direction)
    return (x, y)

def external_forces(t):
    mass = mass(t)
    