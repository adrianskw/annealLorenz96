# annealLorenz96
Testing the limits of VarAnneal for large dimensions, specifically the Lorenz96 model.

Dependencies:
Python with full Scipy stack (the usual when it comes to scientific computing)

To use:

	python makeLorenz.py <value>
Numerically generates a Lorenz system with D=\<value\> dimension, for a fixed number of timesteps M=200, dt_observed = 0.1, and dt_generation=0.001. The resulting files are stored as data/inputs/L96_D\<value\>.dat. Currently, D = 5,10,20,50,100,200,500,1000 have already been generated and are stored in this repository.

	python annealLorenz.py <value>
Anneals the data in data/inputs/L96_D\<value\>.dat 
