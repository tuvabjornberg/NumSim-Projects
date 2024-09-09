import numpy as np 
import scipy

def mass(t):
    if t <= 10:
        return 8 - 0.4 * t
    else:
        return 4

print(mass(30))
print(mass(5))
print(mass(10))

