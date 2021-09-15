# Sequential Design of Adsorption Simulations in Metal-Organic Frameworks![AL_workflow](https://user-images.githubusercontent.com/36941306/133353572-0bbcdaf5-3d7f-463a-9973-d8b935d89132.png)
The process of building an AL model is divided into two steps
1. Generating a prior dataset. 
This step can further be broken two smaller steps.
  (i) The first involves creating a prior input files, which would contain all the features. For a single feature, this file would have pressure.
For a double features, it would have pressure and temperature both.
  We would use the code prior.py to do it, and we can generate the prior in three way — a) Boundary-informed prior, b) Linear-spaced LHS, and c) Log-spaced LHS. For details please refer to the original paper. In the prior.py code, give the necessary conditions, this would pressure limits and temperature limits for the LHS-based priors, and the hand-picked pressure and temperature conditions for boundary informed prior.
  After giving the necesaary conditions, run the code in python and we have a .csv file called 'Prior_test.csv'
(ii) Use build-prior-sh to submit the simulation to a remote computing system. At Notre Dame, the cluster is based on grid-engine. For other clusters, necassary changes need to be made. In the build-prior-sh, you need to input the number of sample point you want to generate (this should be equivalent to the number of points in 'Prior_test.csv'). 

The second step involves conducting the simulation for generating the dataset and then performing data extraction.
2. Conducting Active learning.
The entire theory behing the AL has been highlighted in this picture above. For details please refer to the original paper.
