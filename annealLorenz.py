
"""""""""""""""""""""""""""""""""""""""""""""""""""
Some extra stuff for quality of life improvement
"""""""""""""""""""""""""""""""""""""""""""""""""""
import sys, os, time
import numpy as np

# For code timing
tstart = time.time()

# Parsing the input arguments
if (len(sys.argv)!=4):
    print("Please enter proper arguments for D, dt, L_frac respectively.")
    sys.exit(0)

# Setting the parsed values to D, dt, L_frac
# D         = number of variables, aka dimension of measurement
# dt        = timestep of measurements
# L_frac    = fraction of variables that are measured
D,dt,L_frac  = np.asarray(sys.argv[1:])
D = D.astype(int)
dt = dt.astype(float)
L_frac = L_frac.astype(float)
print "Running annealLorenz for D = "+str(D)+ \
        ", dt = "+str(dt)+", L_frac = "+str(L_frac)

# Disable all print statements
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Enable all print statements
def enablePrint():
    sys.stdout = sys.__stdout__

"""""""""""""""""""""""""""""""""""""""""""""""""""
Begin: Making the data for the Lorenz96 model
"""""""""""""""""""""""""""""""""""""""""""""""""""
from scipy.integrate import odeint

# NOTE:
# Data is generated with a small and constant timestep dt = 0.001
# The output dt is different from the integrating dt that generates the system
# This dt_output/dt_generation ratio is like the measurement "resolution"
# The procedure here is the following:
# 1) Generated a large Lorenz96 with a small timestep dt_gen
# 2) Artificially sample the large data set at a lower resolution timestep dt

# Static Constants NOTE: There's no reason to touch most of these values
# F         = forcing, NOTE: F = 8.17 gives chaotic behavior
# M         = number of timesteps for OUTPUT, NOTE: output array will have M+1 elements
# T         = end time
# t         = array of times for OUTPUT
# dt_gen    = dt that integrates the sytem
# M_gen     = number of timesteps for generated data
# t_gen     = array of times for generated data
# frac      = amount to skip, ie. 1/resolution
# x0        = initial state of the system (equilibrium)
F       = 8.17
M       = 200 # change me in the future
T       = M*dt
t       = np.linspace(0,T,M+1)
dt_gen  = 0.001
M_gen   = int(T/dt_gen)
t_gen   = np.linspace(0, T, M_gen+1)
frac    = int(dt/dt_gen)
x0      = F*np.ones(D)

# Add small perturbation to 1st variable and breaking from equailibrium
x0[0] += 0.01

# Definition of the Lorenz96 model for the odeint function below
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

# Performing the integration, gives us the full data set
x = odeint(Lorenz96, x0, t_gen)

# Creating a sparser array y from the full array x
# y is the OUTPUT array
# x is the generated data
y = np.ones([M+1,D+1])
for i in range(0,M+1):
    for j in range(0,D):
        y[i,j] = x[frac*i,j]

"""""""""""""""""""""""""""""""""""""""""""""""""""
Setting up inputs for VarAnneal
"""""""""""""""""""""""""""""""""""""""""""""""""""

from varanneal import va_ode

# Quick definition of Lorenz96 for VarAnneal
def L96(t,x,k):
    return np.roll(x,1,1)*(np.roll(x,-1,1)-np.roll(x,2,1))-x+k

# Our Constants:
# M_model = number of timesteps presented to the model (M_model < M)
# Rm = Measurement Error (fixed)
# Rf0 = Model Error (initial)
# alpha = Multiplier of Rf (eg. Rf[i+1]=alpha*Rf[i])
# steps = Number of annealing steps to take
M_model     = int(M/2)+1 # we are only assimilating the first half of data
Rm          = 5.0
Rf0         = 1e-6
alpha       = 1.5
steps       = 200

# Lidx = Index of measured variables (automatic)
Lidx  = np.arange(0,D-1,1.0/L_frac).astype(int)
print("Feeding the following dimensions into VarAnneal: "+str(Lidx))

# Pidx = Index of parameters
# beta = Index of Rf (eg. Rf[beta]=Rf0*alpha^beta)
# X0 = Initial paths
# P0 = Initial parameters
# subdata = Subset of time series data to be fed into varanneal
# subtimes = Subset of times to be fed into varanneal
Pidx  = [0]
beta  = np.linspace(0,steps-1,steps)
#X0 = (10.0 * (2*np.random.rand(M_model,D)-1))
X0 = np.ones((M_model,D))
#P0 = np.array([4.0*np.random.rand()+6.0])
P0 = np.array([6])

# Defining a subset of times and data to be fed into the model
t_model = t[0:M_model]
y_model = y[0:M_model,Lidx]

"""""""""""""""""""""""""""""""""""""""""""""""""""
Running VarAnneal (workhorse)
"""""""""""""""""""""""""""""""""""""""""""""""""""
# My annealer initiaization
# All this is in Paul Rozdeba's VarAnneal code
blockPrint()
myannealer = va_ode.Annealer()
myannealer.set_model(L96,D)
myannealer.set_data(y_model,t=t_model)
BFGS_options = {'gtol':1.0e-8, 'ftol':1.0e-8, 'maxfun':1000000, 'maxiter':1000000}

myannealer.anneal(X0, P0, alpha, beta, Rm, Rf0, Lidx, Pidx, dt_model=dt,
                  init_to_data=True, disc='SimpsonHermite', method='L-BFGS-B',
                  opt_args=BFGS_options, adolcID=0)
enablePrint()

# Setting up unique ID for output
ID = "_D="+str(D)+"_dt="+str(dt)+"_Lfrac="+str(L_frac)+".npy"
myannealer.save_paths("./data/outputs/paths"+ID)  # Path estimates
myannealer.save_params("./data/outputs/params"+ID)  # Parameter estimates
myannealer.save_action_errors("./data/outputs/action_errors"+ID)  # Action and individual error terms

print("Annealing done. Run time is "+str(time.time()-tstart)+"seconds.")
