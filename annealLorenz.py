
"""
Some extra functions to block print statements
"""

import sys, os, time
# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

"""
Code starts here
"""

import numpy as np
from varanneal import va_ode

# Quick definition of our Lorenz96 model
def L96(t,x,k):
    return np.roll(x,1,1)*(np.roll(x,-1,1)-np.roll(x,2,1))-x+k

# Importing data from ASCII file
data  = np.loadtxt("L96.dat")
times = data[:,0]

# Our Constants
# M_data = number of timesteps for data
# M_model = number of timesteps presented to the model (M_model < M_data)
# D = Number of variables
# dt = Fixed timestep
# Rm = Measurement Error (fixed)
# Rf0 = Model Error (initial)
# alpha = Multiplier of Rf (eg. Rf[i+1]=alpha*Rf[i])
# steps = Number of annealing steps to take
M_data  = len(times)
M_model = int(M_data/2)+1
D       = 5
dt      = times[1]-times[0]
Rm      = 5.0
Rf0     = 1e-6
alpha   = 1.5
steps   = 100

# Lidx = Index of measured variables (index starts at 1)
# Pidx = Index of parameters
# beta = Index of Rf (eg. Rf[beta]=Rf0*alpha^beta)
# X0 = Initial paths (random)
# P0 = Initial parameters (random)
# subdata = Subset of time series data to be fed into varanneal
# subtimes = Subset of times to be fed into varanneal
#Lidx  = np.linspace(1,D,np.ceil(D*0.4-0.1),dtype=int)
Lidx = [1,4]
Pidx  = [0]
beta  = np.linspace(0,steps-1,steps)
#X0 = (10.0 * (2*np.random.rand(M_model,D)-1))
X0 = np.ones((M_model,D))
#P0 = np.array([4.0*np.random.rand()+6.0])
P0 = 1
subdata  = data[0:M_model,Lidx]
subtimes = times[0:M_model]

blockPrint()
# My annealer initiaization
myannealer = va_ode.Annealer()
myannealer.set_model(L96,D)
myannealer.set_data(subdata,t=subtimes)
BFGS_options = {'gtol':1.0e-8, 'ftol':1.0e-8, 'maxfun':1000000, 'maxiter':1000000}

myannealer.anneal(X0, P0, alpha, beta, Rm, Rf0, Lidx, Pidx, dt_model=dt,
                  init_to_data=True, disc='SimpsonHermite', method='L-BFGS-B',
                  opt_args=BFGS_options, adolcID=0)
enablePrint()
print("Annealing done.")

myannealer.save_paths("paths.npy")  # Path estimates
myannealer.save_params("params.npy")  # Parameter estimates
myannealer.save_action_errors("action_errors.npy")  # Action and individual error terms