#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 14:12:25 2025

@author: emilrasmussen
"""

import steady_state as steady
import numpy as np

def production_function(x,y): 
    return x*y

delta = 1
rho = 100
r = 1
n = 500
tol = 1e-12

steady.solve_model(n, delta, rho, r, production_function,tol)
