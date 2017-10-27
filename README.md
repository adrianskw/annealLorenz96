# annealLorenz96
Testing the limits of VarAnneal applied to Lorenz96 in the case of large dimensions and measurement sparsity.

Dependencies:
Python with full Scipy stack (the usual when it comes to scientific computing)

## To use:

	python annealLorenz.py D dt L_frac
Numerically generates a Lorenz system with dimensions D and timestep dt, for a fixed number of timesteps M=200. The data is immediately fed into VarAnneal with a fraction of measured variables L_frac, which is automatically determined.
