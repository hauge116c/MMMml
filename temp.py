# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt

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

x_initial_guess = 1.5
x_solution = fsolve(func,x_initial_guess)
