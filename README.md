# annealLorenz96
Testing the limits of VarAnneal applied to Lorenz96 in the case of large dimensions and measurement sparsity.

Dependencies:
Python with full Scipy stack (the usual when it comes to scientific computing)

## To use:
	python annealLorenz.py D M dt L_frac
Numerically generates a Lorenz system with dimensions D, number of timesteps M, and timestep size dt. The data is immediately fed into VarAnneal with a fraction of measured variables L_frac, which is automatically determined. D, M, dt, and L_frac have to be input in the command line and the output files are labeled accordingly. This allows for easy scripting.

## Comments:
It seems that for certain values of M (eg. 50), the code doesn't work due to the some laziness on my part. I choose values of M=10,20,40,100 etc.
The path is initialized as an array of ones, and the parameter is initialized at 6.0. Also, integer parameters will caused rounding issues.
