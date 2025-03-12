#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 08:38:21 2025

@author: emilrasmussen
"""

from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pandas as pd
from scipy import integrate
from mpl_toolkits.mplot3d import Axes3D

#%% Define path 
home = str(Path.home())
path = home + "/Documents/GitHub/MMMml"
plotpath = path + "/Plots"

#%% Import data
import_male = pd.read_csv(path+"/income_distribution_male.csv").to_numpy(copy=True)
import_female = pd.read_csv(path + "/income_distribution_female.csv").to_numpy(copy=True)

#%% Reshaping data

#%% Parameters for the model
#Creating a dictionary for different values
p = dict()
#Nash-bargaining power
p['c_beta']=0.5
#Discout rate, r
p['c_r']=0.05
#Seperation rate
p['c_delta']=0.1
# Matching function
p['c_lambda_m']=1
p['c_lambda_f']=1
# Add outside options for unmatched males and females
p['c_m'] = 0.0  # Default outside option for males (adjust as needed)
p['c_f'] = 0.0  # Default outside option for females (adjust as needed)

# Grid points for men
print(np.shape(import_male))
p['xmin'] = import_male[0,0]
print('Lowest income grid points:',p['xmin'])
p['xmax'] = import_male[49,0]
print('Highest income grid points:',p['xmax'])
p['xstep'] = import_male[1,0]-import_male[0,0]
print('Stepsize:',p['xstep'])
# Type space
p['typespace'] = import_male[:,0]
#p['typespace'] = p['typespace_n']/np.min(p['typespace_n']) #Normalized lowest value to 1
xgrid = p['typespace'].ravel() #Helps you with the dimensions for a vector
ygrid = p['typespace'].ravel()

p['n_types'] = 50
n = p['n_types']
p['male_inc_dens'] = import_male[:,1] #Take the entire row or coloumn,
p['female_inc_dens'] = import_female[:,1]


#%% Production function
def production_function(x, y): 
    return x * y

#%% Numerical integration
def integrate_uni(values,xstep):
    """ Integrates the grid with the specified values of the interval [x0,x1]"""
    #spacing = x1/values.size
    copy = np.copy(values)
    copy.shape = (1,values.size)
    return integrate.simpson(copy,dx=xstep)

def integrate_red(matrix,result,xstep):
    n = matrix.shape[0]
    if result =='male':
        inner = np.zeros((1,n))
        for i in range(0,n):
            inner[0,i] = np.squeeze(integrate_uni(matrix[:,1],xstep))
    elif result=='female':
        inner = np.zeros((n,1))
        for i in range(0,n):
            inner[i,0] = np.squeeze(integrate_uni(matrix[i,:],xstep))
    return inner


def flow_update (dens_e, dens_u_o, alpha, c_delta, result, c_lambda_m, xstep): #"o" betyder other gender så modsat af en selv
    int_u_o = integrate_red(dens_u_o * alpha, result, xstep) #integrate_red integrerer over en matrix i stedet for en vektor. Vores integrate uni ovenfor integrerer over en vektor.
    int_u_o.shape = dens_e.shape
    return c_delta*dens_e / (c_delta+c_lambda_m*int_u_o)


integrate_uni(p['male_inc_dens'],p['xstep'])

#Normalize densities
#Density function for all agents
e_m = p['male_inc_dens'] / integrate_uni(p['male_inc_dens'],p['xstep'])
e_m.shape = (1,p['n_types'])
e_f = p['female_inc_dens'] / integrate_uni(p['female_inc_dens'],p['xstep'])
e_f.shape = (p['n_types'],1)

# Extract single densities // endgoenous U-densities
u_m = np.ones((1,p['n_types']))
u_f = np.ones((p['n_types'],1))

# Production function

def production_function(x,y): 
    return x*y

# Flow utilies for married

c_xy = np.zeros([p['n_types'],p['n_types']])
for xi in range(len(xgrid)):
    for yi in range(len(ygrid)):
        c_xy[xi,yi] = production_function(xgrid[xi],ygrid[yi])

c_x = np.zeros((1,p['n_types']))
c_y = np.zeros((p['n_types'],1))

for xi in range(p['n_types']):
    c_x[0,xi] = xgrid[xi]
    
for yi in range(p['n_types'],1):
    c_y[yi,0] = ygrid[yi]   

values_s_m = c_x
values_s_f = c_y

maxiter = 1000
tolerance = 1e-6  # Convergence tolerance
converged = False
#%% From class, but not completed
#for iter in range(maxiter):
#    
#        alpha= np.ones((p['n_types'], p['n_types']))
#        int_U_m = integrate_uni(u_m, p['xstep']) #er i tvivl om der skal stå p['xmin'], p['xmax'] eller bare xmin, xmax her? Same for linjen lige under.
#        int_U_f = integrate_uni(u_f, p['xstep'])
#        
#        u_m_1 = flow_update(e_m, u_f, alpha, p['c_delta'], 'male', p['c_lambda_m'], p['xstep'])
#       u_m_1 = flow_update(e_f, u_f, alpha, p['c_delta'], 'male', p['c_lambda_m'], p['xstep'])


#%% From Grok, tried to see if that could complete the code
for iter in range(maxiter):
    # Store old values for convergence check
    u_m_old = u_m.copy()
    u_f_old = u_f.copy()
    
    # Calculate matching probabilities (alpha)
    # This could be based on the production function and income matching
    alpha = np.zeros((p['n_types'], p['n_types']))
    for i in range(p['n_types']):
        for j in range(p['n_types']):
            # Simple positive assortative matching based on production
            alpha[i,j] = c_xy[i,j] / (c_xy[i,:].sum() + c_xy[:,j].sum())
    
    # Update unemployed densities using flow equations
    # Flow out = Flow in (steady state)
    int_u_f = integrate_red(u_f * alpha, 'male', p['xstep'])  # Integrated over females for each male type
    int_u_m = integrate_red(u_m * alpha, 'female', p['xstep'])  # Integrated over males for each female type
    
    # Update densities (flow balance equations)
    u_m_new = p['c_delta'] * e_m / (p['c_delta'] + p['c_lambda_m'] * int_u_f)
    u_f_new = p['c_delta'] * e_f / (p['c_delta'] + p['c_lambda_f'] * int_u_m)
    
    # Check convergence
    diff_m = np.max(np.abs(u_m_new - u_m))
    diff_f = np.max(np.abs(u_f_new - u_f))
    
    # Update values
    u_m = u_m_new.copy()
    u_f = u_f_new.copy()
    
    if max(diff_m, diff_f) < tolerance:
        converged = True
        print(f"Converged after {iter+1} iterations")
        break
    
if not converged:
    print("Warning: Maximum iterations reached without convergence")
    
    # This iteration has to stop at some point
# Write d.11 + d.12 and d.19 from Appendix // See slides for Marriage p. 21 'Equillibrium'

# To extract a value from the dictionary, you can write: p['xmin'] // p['variable_name_you_want']

# Add z-grid for love shocks (assume uniform [0, 1] for simplicity)
Z_max = 1.0  # Maximum love shock
n_z = 20     # Number of grid points for z
z_grid = np.linspace(0, Z_max, n_z)  # Discretize z
dz = Z_max / (n_z - 1)  # Step size for z

# Cumulative distribution function G(z) for uniform [0, 1]
def G(z):
    return z / Z_max  # CDF for uniform [0, Z_max]

# Initialize variables
maxiter = 1000
tolerance = 1e-6
converged = False

# Initial values
u_m = np.ones((1, p['n_types']))  # Male unemployed density
u_f = np.ones((p['n_types'], 1))  # Female unemployed density
s_m = np.zeros((1, p['n_types']))  # Male surplus
s_f = np.zeros((p['n_types'], 1))  # Female surplus

for iter in range(maxiter):
    # Store old values for convergence check
    u_m_old = u_m.copy()
    u_f_old = u_f.copy()
    s_m_old = s_m.copy()
    s_f_old = s_f.copy()
    
    # Calculate matching probabilities (alpha) - positive assortative matching
    alpha = np.zeros((p['n_types'], p['n_types']))
    for i in range(p['n_types']):
        for j in range(p['n_types']):
            alpha[i, j] = production_function(xgrid[i], ygrid[j]) / (production_function(xgrid[i], ygrid).sum() + production_function(xgrid, ygrid[j]).sum())
    
    # Integrate over y for u_m(x) (Equation 1)
    int_u_f_alpha = np.zeros((1, p['n_types']))
    for i in range(p['n_types']):
        values = u_f[:, 0] * alpha[i, :]  # u_f(y) * α(x, y)
        int_u_f_alpha[0, i] = integrate.simpson(values, dx=p['xstep'])
    u_m_new = e_m / (1 + (p['c_delta'] / p['c_lambda_m']) * int_u_f_alpha)
    
    # Integrate over x for u_f(y) (Equation 2)
    int_u_m_alpha = np.zeros((p['n_types'], 1))
    for j in range(p['n_types']):
        values = u_m[0, :] * alpha[:, j]  # u_m(x) * α(x, y)
        int_u_m_alpha[j, 0] = integrate.simpson(values, dx=p['xstep'])
    u_f_new = e_f / (1 + (p['c_delta'] / p['c_lambda_f']) * int_u_m_alpha)
    
    # Calculate s_m(x) (Equation 3) - Integrate over y and z
    s_m_new = np.zeros((1, p['n_types']))
    for i in range(p['n_types']):
        total = 0
        for j in range(p['n_types']):
            for k in range(n_z):
                z = z_grid[k]
                surplus = max(z + production_function(xgrid[i], ygrid[j]) - s_f[j, 0], 0)  # max{ z + C(x,y) - s_f(y), 0 }
                total += surplus * u_f[j, 0] * (G(z + dz) - G(z))  # Integrate over z using trapezoidal rule approximation
        s_m_new[0, i] = (p['c_m'] + (p['c_delta'] / (p['c_r'] + p['c_delta'])) * total) / (1 + (p['c_lambda_m'] ** (1 - p['c_beta'])) / (p['c_r'] + p['c_delta']) * u_f.sum())
    
    # Calculate s_f(y) (Equation 4) - Integrate over x and z
    s_f_new = np.zeros((p['n_types'], 1))
    for j in range(p['n_types']):
        total = 0
        for i in range(p['n_types']):
            for k in range(n_z):
                z = z_grid[k]
                surplus = max(z + production_function(xgrid[i], ygrid[j]) - s_m[0, i], 0)  # max{ z + C(x,y) - s_m(x), 0 }
                total += surplus * u_m[0, i] * (G(z + dz) - G(z))  # Integrate over z
        s_f_new[j, 0] = (p['c_f'] + (p['c_delta'] / (p['c_r'] + p['c_delta'])) * total) / (1 + (p['c_lambda_f'] ** (1 - p['c_beta'])) / (p['c_r'] + p['c_delta']) * u_m.sum())
    
    # Update values
    u_m = u_m_new.copy()
    u_f = u_f_new.copy()
    s_m = s_m_new.copy()
    s_f = s_f_new.copy()
    
    # Check convergence
    diff_u_m = np.max(np.abs(u_m - u_m_old))
    diff_u_f = np.max(np.abs(u_f - u_f_old))
    diff_s_m = np.max(np.abs(s_m - s_m_old))
    diff_s_f = np.max(np.abs(s_f - s_f_old))
    
    if max(diff_u_m, diff_u_f, diff_s_m, diff_s_f) < tolerance:
        converged = True
        print(f"Converged after {iter + 1} iterations")
        break
    
if not converged:
    print("Warning: Maximum iterations reached without convergence")

# Update global variables for further use
p['u_m'] = u_m
p['u_f'] = u_f
p['s_m'] = s_m
p['s_f'] = s_f

#%% Plot
# Create a meshgrid for x and y
X, Y = np.meshgrid(p['typespace'], p['typespace'])

# Calculate the production function values for the meshgrid
Z = production_function(X, Y)  # Using your current production_function(x, y) = x * y

# Create the 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

# Add a color bar
plt.colorbar(surf, label='Production Output')

# Set labels
ax.set_xlabel('Male Income (x)')
ax.set_ylabel('Female Income (y)')
ax.set_zlabel('Production Output')

# Set title
ax.set_title('Estimated Production Function')

# Adjust the view angle to match your image (you can tweak these values)
ax.view_init(elev=20, azim=45)

plt.show()



                            