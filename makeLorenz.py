# This code is from the Wikipedia Lorenz96 page with slight modifications
# It generates a Lorenz96 model according to the given parameters

from scipy.integrate import odeint
import numpy as np

# NOTE:
# Data is generated with a small and constant timestep dt = 0.001
# The output dt is different from the integrating dt that generates the system
# This dt_output/dt_generation ratio can be thought of as the "resolution"
# The procedure here is the following:
# 1) Generated a large Lorenz96 will a small timestep dt_gen
# 2) Artificially sample the large data set at a lower resolution timestep dt
# 3) Output the sparser data set accordingly

# Our constants:
# D  = number of variables, aka dimension of measurement
# F  = forcing, NOTE: F = 8.17 gives chaotic behavior
# dt = dt that the code outputs, aka measurement times
# M  = number of timesteps, NOTE: output will have M+1 timesteps
# T  = end time
# t  = array of output times
D  = 1000
F  = 8.17
dt = 0.1
M  = 200
T  = M*dt
t  = np.linspace(0,T,M+1)

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
# end Lorenz96(x,t)

# dt_gen    = dt that integrates the sytem, should not be touched
# M_gen     = number of timesteps for the generated data
# t_gen     = array of times
# frac      = amount to skip
# x0        = initial state of the system (equilibrium)
dt_gen  = 0.001
M_gen   = int(T/dt_gen)
t_gen   = np.linspace(0, T, M_gen+1)
frac    = int(dt/dt_gen)
x0      = F*np.ones(D)

# Add small perturbation to 1st variable, inducing chaotic behaviour
x0[0] += 0.01

# Performing the integration, gives us the full data set
x = odeint(Lorenz96, x0, t_gen)

# Creating a sparser array y from the full array x
y = np.ones([M+1,D+1])
for i in range(0,M+1):
    for j in range(0,D+1):
        if(j==0):
            y[i,0] = t[i]
        else:
            y[i,j] = x[frac*i,j-1]

# Writing Lorenz96 data to a data file
s = "./data/inputs/L96_D"+str(D)+".dat"
np.savetxt(s,y,fmt="%6f",delimiter='\t')
print("Lorenz96 model made with:")
print("D  = "+str(D))
print("F  = "+str(F))
print("dt = "+str(dt))
print("M  = "+str(M))
