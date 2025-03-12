#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 09:45:08 2025

@author: emilrasmussen
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%% Importing stuff
from scipy.optimize import fsolve
from scipy.optimize import least_squares
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

#%% Define path for plots
home = str(Path.home())
path = home + "/Documents/GitHub/MMMml"
plotpath = path + "/Plots"


#%% Defining parameters
b = 0.4         # Benefit level
p = 1           # Price, standardized
r = 0.0025      # Interest rate
c = 0.21        # Vacancy cost
delta = 0.01    # Job destruction rate
beta = 0.5      # Bargaining power
alpha_0 = 0.55  # Level in matching function
alpha_1 = 0.5   # Elasticity of matching function w.r.t. unemployment

#%% Initial guess
initial_guess = [0.1,0.5,0.5]
initial_guess_array = np.array(initial_guess)
bottom_bound = [0,0,0]
upper_bound = [1,1,1]

#%% Define function to solve DMP steady state
# def DMP_steady_state(X,*parameters):
    # parameters = (b,p,r,delta,beta,alpha_0,alpha_1, c)
    # Unpack parameters
    # b,p,r,delta,beta,alpha_0,alpha_1, c = parameters
    
def DMP_steady_state(X,b,p,r,delta,beta,alpha_0,alpha_1, c):
    w = X[0]
    u = X[1]
    v = X[2]
    
    # Labor market thightness
    theta = v / u
    
    # Vacancy filling rate, q
    def q_theta(theta):
        return alpha_0 * theta**(-alpha_1)
    
    # Job finding rate, \lambda
    def lambda_theta(theta):
        return alpha_0 * theta**(alpha_1)
    
    # Vector for storage of results
    out = np.zeros_like(X)
    
    # Beveridge equation
    bev = u - delta / (lambda_theta(theta) + delta)
    out[0] = bev
    # Free entry condition
    free_entry = (p-w) / (r+delta) - c / q_theta(theta)
    out[1] = free_entry
    
    # Nash bargaining
    nash = w - (b + beta*(p-b)+beta*c*theta)
    out[2] = nash
    
    return out

#%% Solving the system of equations
solution = fsolve(DMP_steady_state, initial_guess_array, args=(b, p, r, delta, beta, alpha_0, alpha_1, c))
solution_1 = least_squares(DMP_steady_state,initial_guess_array, args=(b, p, r, delta, beta, alpha_0, alpha_1, c), bounds=(bottom_bound,upper_bound))
print(solution_1)
w_star_1 = solution_1.x[0]
u_star_1 = solution_1.x[1]
v_star_1 = solution_1.x[2]
w_star, u_star, v_star = solution
theta_star = v_star / u_star

# Print the result
print("Steady state wage:", w_star)
print("Steady state unemployment:", u_star)
print("Steady state vacancies:", v_star)
print("Steady state labor market tightness:", theta_star)

#%%Plot data 1
theta_range = np.arange(0,7.5,0.5)
wage_curve = b + beta*(p-b)+beta*c*theta_range # Nash bargaining, solved for w
job_creation = p - c / (alpha_0 * theta_range**(-alpha_1)) *(r+delta)# Free entry, solved for w



#%%% Plot figure, Wage curve and Job-creation curve

plt.figure()
plt.plot(theta_range, wage_curve, label="Wage Curve", color="blue")
plt.plot(theta_range, job_creation, label="Job Creation Curve", color="orange")
plt.axvline(x=theta_star, color="red", linestyle="--", label="θ* (steady state)")
plt.scatter(theta_star, w_star, color="black", label="Steady-state (w*, θ*)")
plt.xlabel("Labor Market Tightness (θ)")
plt.ylabel("Wage (w)")
plt.title("Wage Curve and Job Creation Curve")
plt.legend()
plt.grid()
plt.show()

#%%Plot data 2
#def v_value (x):    
#    
#    w_value = b+beta*(p-b)+beta*c*theta_star
#    
#    v_hat = u_range*((p-w_value/(r+delta))*alpha_0/c)**(1/alpha_1)
#    
#    return v_hat
#
#u_range = np.arange(0,10,0.5)
#beve_curve = u_range *((delta/alpha_0)*((1-u_range)/u_range))**(1/(1-alpha_1))
#job_creation_curve = v_value(u_range)

#%%% Plot figure, Beveridge curve and Job-creation curve

#plt.figure()
#plt.plot(u_range, job_creation_curve, label="Job creation, color="blue")
#plt.plot(u_range, beve_curve, label="Beveridge curve", color="orange")
#plt.axvline(x=theta_star, color="red", linestyle="--", label="θ* (steady state)")
#plt.scatter(theta_star, w_star, color="black", label="Steady-state (w*, θ*)")
#plt.xlabel("Labor Market Tightness (θ)")
#plt.ylabel("Wage (w)")
#plt.title("Wage Curve and Job Creation Curve")
#plt.legend()
#plt.grid()
#plt.show()