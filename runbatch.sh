#!/bin/bash
# Quick scripting for doing a batch of annealLorenz
# You can just set the values of interest and go do something else
# Remember, annealLorenz takes in 4 arguments:
# D = number of variables
# M = number of timesteps
# dt = size of timestep
# L_frac = fraction of D that is "measured"
# run the code as:
#   bash runbatch.sh
# All output is appended to the end of a log file
# Runtime is recorded in the log file
python annealLorenz.py 5 100 0.2 0.4 >> logs/out.log
echo "DONE 1"
python annealLorenz.py 5 40 0.5 0.4 >> logs/out.log
echo "DONE 2"
