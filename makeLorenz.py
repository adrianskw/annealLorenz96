# This code is from the Wikipedia Lorenz96 page with some slight modifications
# It generates a Lorenz96 model according to the given parameters

from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np

# Our constants:
# D = number of variables
# F = forcing
# T = end time
# dt = constant timestep
# M = number of timesteps
D  = 5
F  = 8.17
T  = 100.0
dt = 1
M  = int(T/dt)

def Lorenz96(x,t):
  # compute state derivatives
  d = np.zeros(D)
  # first the 3 edge cases (initial conditions)
  d[  0] = (x[1] - x[D-2]) * x[D-1] - x[  0]
  d[  1] = (x[2] - x[D-1]) * x[  0] - x[  1]
  d[D-1] = (x[0] - x[D-3]) * x[D-2] - x[D-1]
  # then the general case
  for i in range(2, D-1):
      d[i] = (x[i+1] - x[i-2]) * x[i-1] - x[i]
  # return the state derivatives + forcing term
  return d+F

# initial state (equilibrium)
x0 = F*np.ones(D)
# add small perturbation to 1st variable
x0[0] += 0.01
# list of times
t = np.linspace(dt, T, M)

# Performing the integration
x = odeint(Lorenz96, x0, t)

# Modifying our vector so that time is displayed in the front
y = np.ones([M,D+1])
for i in range(0,M):
    for j in range(0,D+1):
        if(j==0):
            y[i,j] = t[i]
        else:
            y[i,j] = x[i,j-1]

# writing Lorenz96 data to a data file
np.savetxt("./data/inputs/L96.dat",y,fmt="%6f") # can add delimiters = ', '
