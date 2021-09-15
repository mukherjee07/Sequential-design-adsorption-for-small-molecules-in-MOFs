'''
Author: Krish-the evergreen-mukherjee
Advisor: Dr. Yamil Colon, University of Notre Dame
Objective: The objective of this code is to create a Prior dataset for active updating for later stages. We are going to employ Latin hypercube sampling (LHS) to achieve this.
'''

#importing libraries
import numpy as np
import pandas as pd
import math

#importing the latin-hypercube sampling multi-dimensional uniformity 
import lhsmdu 

#importing operating system library
import os

#Creating a prior data file for printing sample datapoints
FILENAME='Prior_test.csv'
os.system("rm -f "+str(FILENAME))
os.system("touch "+str(FILENAME))
os.system("echo Pressure_in_Pascal Temperature_in_K >> "+str(FILENAME))

T = [100,150,200,250,300]
P = [0.1,1,10,100,1000,10000,100000,10000000,20000000,30000000]

for i in range(len(T)):
	for j in range(len(P)):
		os.system("echo "+str(P[j])+" "+str(T[i])+" >> "+str(FILENAME))
