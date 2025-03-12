# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%% Importing stuff
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

#%% Define path for plots
home = str(Path.home())
path = home + "/Documents/GitHub/MMMml"
plotpath = path + "/Plots"


#%% Defining the function
def function(x): 
    """
    This is a function that takes the input x to the power of 3, f(x)=x^3

    Parameters
    ----------
    x : Parameter

    Returns
    -------
    y : The root for 5 - x^3 = 0.

    """
    y = 5-x**3
    return y




func = lambda x : 5 - x**3

#%% Solve the function for the initual guess
#np.power(base,exponent) --> To call the power of a number.
x_initial_guess = 1.5
x_solution = fsolve(func,x_initial_guess)

#%%
x_range = np.arange(-2,4,0.1)
y_output = function(x_range)
plt.plot(x_range,y_output)
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Plot of $5 - x^3$ with Root Highlighted')
plt.legend()
plt.grid(True)
plt.show()

#%% Another plot
fig1, ax1 = plt.subplots()
ax1.plot(x_range, y_output)
ax1.set_title('Root for f(x)=x^3')
#ax1.xlabel('x')
#ax1.ylabel('f(x)')
#ax1.title('Plot of $5 - x^3$ with Root Highlighted')
ax1.legend()
ax1.grid(True)
fig1.savefig(plotpath + "/function.pdf")