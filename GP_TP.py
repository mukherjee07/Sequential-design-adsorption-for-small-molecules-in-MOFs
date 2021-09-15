#!/usr/bin/env python
# coding: utf-8

#importing important libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import math

#importing the ML libraries
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, WhiteKernel 
from sklearn.gaussian_process.kernels import ConstantKernel as C

#Reading the dataset
df = pd.read_csv('prior_Sample.csv',delimiter=',')
df2 = pd.read_csv('complete.csv',delimiter=',')
#print(df.head())

#sorting out the data
#df = df.sort_values(by=['Pressure'], ascending='True')
#print(df.head(n=40))

#Substitute a data row with some non-zero value if methane uptake for that row zero, this is to avoid log(0) = NaN error
#df['Uptake_for_Cu-BTC-CH4(300K)'].replace(to_replace=0, value=0.00001)
#print(df.head(n=40))

#print(df)
#Unseen array
P_test = np.atleast_2d(np.linspace(1e-6,300,50)).flatten().reshape(-1,1)
T_test = np.atleast_2d(np.linspace(100,300,40)).flatten().reshape(-1,1)

''' 
FOR generating Prior dataset for comparision 
for i in range(len(X_test)):
	os.system("echo "+str(X_test[i])+" >> Prior.csv")
'''

#Reading the data
p = df.iloc[:,0].values
t = df.iloc[:,1].values
y = df.iloc[:,2].values
#Taking the error data as well
e = df.iloc[:,3].values

#from complete-original dataset
p2 = df2.iloc[:,0].values
t2 = df2.iloc[:,1].values
y2 = df2.iloc[:,2].values

#Replacing y if some y value in zero
for i in range(len(y)):
	if (y[i] == 0):
		y[i] = 0.0001

#For y2
for i in range(len(y2)):
	if (y2[i] == 0):
		y2[i] = 0.0001

#Transforming 1D arrays to 2D
p = np.atleast_2d(p).flatten().reshape(-1,1)
t = np.atleast_2d(t).flatten().reshape(-1,1)
y = np.atleast_2d(y).flatten()

p_true = p
t_true = t
y_actual = y
#before the transition
#print(x,y)

#converting P to bars
p = p/(1.0e5)

#Taking logbase 10 of the input vector
p = np.log10(p)
t = np.log10(t)
y = np.log10(y)

#print(len(x),len(y))
#Taking the log of X_test
P_test = np.log10(P_test)
T_test = np.log10(T_test)

#Extracting the mean and std. dev for P_test
p_m = np.mean(P_test)
p_std = np.std(P_test,ddof=1)

#Extracting the mean and std. dev for T_test
t_m = np.mean(T_test)
t_std = np.std(T_test,ddof=1)

#Standardising p,t and y in log-space
p_s = (p - p_m)/p_std
t_s = (t - t_m)/t_std

#Standardising X_test in log-space
P_test = (P_test - p_m)/p_std
T_test = (T_test - t_m)/t_std

#Initializing scaled down training and prediction set
#X_test = np.zeros((len(P_test)*len(T_test),2))
x_s = np.zeros((len(p_s),2))

#Filling all the data in training and prediction set
for i in range(len(p_s)):
	for k in range(2):
		#Inserting pressure for the first column
		if k == 0:
			x_s[i,k] = p_s[i]
		else:
		#Inserting temperature for the second column
			x_s[i,k] = t_s[i]
'''
for i in range(len(P_test)):
	for j in range(len(T_test)):
		for k in range(2):
			#Inserting pressure for the first column
			if k == 0:
				X_test[i+j,k] = P_test[i]
			else:
			#Inserting temperature for the second column
				X_test[i+j,k] = T_test[j]
'''
#new definition of X_test
X1, X2 = np.meshgrid(P_test, T_test,indexing='ij')
#Testing the meshgrid function
'''
for i in range(len(X1)):
	for j in range(len(X2)):
		print(X1[i],X2[j])

print(X1,X2)
'''
#print("The type and size of X1 is:",type(X1),len(X1))
X_test = np.vstack((X1.flatten(), X2.flatten())).T


#print(x_s,y)
#Building the GP regresson 
# Instantiate a Gaussian Process model
#kernel = C(1.0, (1e-3, 1e3)) * RationalQuadratic(10, (1e-2, 1e2))
kernel = RationalQuadratic(length_scale=50, alpha=0.5,length_scale_bounds=(1e-13,1e13),alpha_bounds=(1e-13,1e13))+ RationalQuadratic(length_scale=50, alpha=0.5,length_scale_bounds=(1e-13,1e13),alpha_bounds=(1e-13,1e13))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100, normalize_y=True)

#print(type(x_s),type(y.T))
#print(y.T)
#Fitting our normalized data to the GP model
gp.fit(x_s,y.T)
'''
count = 0
for i in range(len(X_test)):
	P_trial = (X_test[i,0]*p_std) + p_m
	P_trial = 10**(P_trial)
	P_trial = 1e5*(P_trial)
	T_trial = (X_test[i,1]*t_std) + t_m
	T_trial = 10**(T_trial)
	print(P_trial,T_trial)
	count += 1
print(count)
'''
# Make the prediction on the test data (ask for MSE as well)
y_pred, sigma = gp.predict(X_test, return_std=True)

Params = gp.get_params()
#print(Params)
#print(y_pred,sigma)
rel_error = np.zeros(len(sigma))

#finding the relative error—
for i in range(len(sigma)):
    rel_error[i] = abs(sigma[i]/abs(y_pred[i]))

#define the limit for uncertainty
lim = 0.02
Max = np.amax(rel_error)
index = np.argmax(rel_error)

#transforming the index to original pressure point
#X_test = (X_test*x_std) + x_m
#X_test = 10**(X_test)
#print(X_test,10**(y_pred),rel_error)

#checking the whether the maximum uncertainty is less than out desired limit
if (Max >= lim):
	Data = str(X_test[index])
	Data = Data.replace("[","")
	Data = Data.replace("]","")
	#Doing the inverse transform now
	X_test[index,0] = (X_test[index,0]*p_std) + p_m
	X_test[index,0] = 10**(X_test[index,0])
	X_test[index,0] = 1e5*(X_test[index,0])
	X_test[index,1] = (X_test[index,1]*t_std) + t_m
	X_test[index,1] = 10**(X_test[index,1])
	print(X_test[index,0],X_test[index,1])
	print("NOT_DONE ")
	print(rel_error[index])
else:
	print(index)
	print("DONE")

y_pred = 10**y_pred
#for i in range(len(y_pred)):
#	print(y_pred[i])
#print(np.shape(X_test),np.shape(y_actual))
#X_test[:,0] = (X_test[:,0]*p_std) + p_m
#X_test[:,0] = 10**(X_test[:,0])
#X_test[:,0] = 1e5*(X_test[:,0])
#X_test[:,1] = (X_test[:,1]*t_std) + t_m
#X_test[:,1] = 10**(X_test[:,1])

#Printing the output for better comparision
'''
for i in range(len(P_test)):
	for j in range(len(T_test)):
		#print(X_test[i+j],y_pred[i+j])
'''

'''
#doing the inverse transform for y
y_pred = sc_y.inverse_transform(y_pred)
y_pred = 10**y_pred
# Doing the inverse transform for X
X_test = sc_X.inverse_transform(X_test)
X_test = 10**X_test

print(X_test,y_pred)
#Transforming the uncertainty
print(len(sigma))
sigma_y = np.zeros(len(sigma))
scaled_rel_error = np.zeros(len(sigma))
for i in range(len(sigma)):
    sigma_y[i] = y_pred[i]*(np.log(10)*sigma[i])
    scaled_rel_error[i] = sigma_y[i]/y_pred[i]
print(np.shape(X_test),np.shape(y_pred),np.shape(sigma_y))
#scaled_rel_error = y_pred/sigma_y
'''
'''
rel_error = 100*(rel_error)

#defining relative root mean square error
rrmse = np.zeros(len(X_test))
#rrmse = 0
for i in range(len(rel_error)):
	rrmse[i] = rrmse[i] + (((y_pred[i] - y2[i])**2)/y2[i]**2)
rrmse = 100*(np.sqrt(rrmse))/len(rel_error)
#print(rrmse)
'''
## For plotting our figure
'''
marker_size = 55
fig1=plt.figure()
fig1.set_figwidth(12)
fig1.set_figheight(8)
plt.rc('xtick',labelsize=18)
plt.rc('ytick',labelsize=18)
plt.rcParams.update({'errorbar.capsize':3})
plt.plot(X_test,y_pred, marker = "H",color="r",Linestyle="solid",label="Predicted GPR-based Cu-BTC MOF uptake")
plt.errorbar(X_test, y_pred,  yerr = rrmse,color = 'k')
plt.scatter(x_true,y_actual, marker = "D", color = "orange",Linestyle="solid",label="Prior data for Cu-BTC MOF uptake")
plt.plot(x2,y2, marker = "*", color = "b", Linestyle = "-.",label="original complete Cu-BTC MOF uptake")
#plt.scatter(X_test,y_test, marker = "8",color="b",Linestyle="--",label="original Cu-BTC MOF uptake")
#plt.errorbar(X_test,y_pred,yerr=sigma_y,label="standard devation",ecolor="k",fmt='r')
#plt.errorbar(X,y_pred,yerr=sigma,label="standard deviation",ecolor="k",fmt='o')
plt.xlabel("Pressure in Pascal ", fontsize = "20")
plt.title("$CH_4$ adsorption isotherm with a 4-point prior sampled from LHS for Cu-BTC MOF",fontsize = "20")
plt.grid(b=True,which='minor',color='#999999',linestyle='-',alpha=0.2)
#plt.ylabel("$CO_2$ adsorption in molecules/cylinder vol.($\AA^3$)", fontsize = "20")
plt.ylabel("$CH_4$ adsorption in molecules/unit cell", fontsize = "20")
plt.legend(loc="best",prop={'size': 18})
plt.savefig('LHS_4pt.png',dpi=800)
plt.show()
'''

'''
X_test=np.log10(X_test)
x_true = np.log10(x_true)
fig2=plt.figure()
fig2.set_figwidth(12)
fig2.set_figheight(8)
plt.rc('xtick',labelsize=18)
plt.rc('ytick',labelsize=18)
plt.rcParams.update({'errorbar.capsize':3})
plt.plot(X_test,y_pred, marker = "H",color="r",Linestyle="solid",label="Predicted GPR-based Cu-BTC MOF uptake")
plt.plot(X_test,y2,color = 'k')
plt.scatter(x_true,y_actual, marker = "D", color = "orange",Linestyle="solid",label="Prior data for Cu-BTC MOF uptake")
#plt.scatter(X_test,y_test, marker = "8",color="b",Linestyle="--",label="original Cu-BTC MOF uptake")
#plt.errorbar(X_test,y_pred,yerr=sigma_y,label="standard devation",ecolor="k",fmt='r')
#plt.errorbar(X,y_pred,yerr=sigma,label="standard deviation",ecolor="k",fmt='o')
plt.xlabel("Pressure in Pascal ", fontsize = "20")
plt.title("$CH_4$ adsorption isotherm with a 4-point prior sampled from LHS for Cu-BTC MOF",fontsize = "20")
plt.grid(b=True,which='minor',color='#999999',linestyle='-',alpha=0.2)
#plt.ylabel("$CO_2$ adsorption in molecules/cylinder vol.($\AA^3$)", fontsize = "20")
plt.ylabel("$CH_4$ adsorption in molecules/unit cell", fontsize = "20")
plt.legend(loc="best",prop={'size': 18})
#plt.savefig('LHS_4pt.png',dpi=800)
plt.show()
'''
'''
'''
'''
fig2=plt.figure()
plt.rc('xtick',labelsize=18)
plt.rc('ytick',labelsize=18)
plt.rcParams.update({'errorbar.capsize':2})
plt.scatter(rel_error,scaled_rel_error, marker = "8",color="r",Linestyle="solid",label="Predicted GPR-based Cu-BTC MOF uptake")
plt.xlabel("relative error ", fontsize = "20")
plt.title("errors",fontsize = "20")
plt.grid(b=True,which='minor',color='#999999',linestyle='-',alpha=0.2)
#plt.ylabel("$CO_2$ adsorption in molecules/cylinder vol.($\AA^3$)", fontsize = "20")
plt.ylabel("relative error (scaled back)", fontsize = "20")
plt.legend(loc="best",prop={'size': 18})
plt.show()
'''
#print(np.shape(scaled_rel_error),np.shape(rel_error))
