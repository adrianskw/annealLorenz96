# annealLorenz96
Testing the limits of VarAnneal applied to Lorenz96 in the case of large dimensions and measurement sparsity.

Dependencies:
Python with full Scipy stack (the usual when it comes to scientific computing)

## To use:
	python annealLorenz.py D M dt L_frac
Numerically generates a Lorenz system with dimensions D, number of timesteps M, and timestep size dt. The data is immediately fed into VarAnneal with a fraction of measured variables L_frac, which is automatically determined. D, M, dt, and L_frac have to be input in the command line and the output files are labeled accordingly. This allows for easy scripting.
