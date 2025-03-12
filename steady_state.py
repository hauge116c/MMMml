#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 14:16:12 2025

@author: emilrasmussen
"""
import numpy as np
import sys

def solve_model(n, delta, rho, r, production_function, tol):
    theta = rho / (2*(r+delta))
    grid = np.linspace(1/n/2, 1-1/n/2, n)
    l_density = 1

    alphas = np.ones([n,n])
    u_density = np.repeat(0.,n)

    payoffs = np.empty([n,n])
    for i in range(n):
        x = grid[i]
        for j in range(n):
            y = grid[j]
            payoffs[i,j] = production_function(x,y)

    keep_iterating = True

#Main loop
    while keep_iterating:
        e = sys.float_info.max
        u_prev = u_density
        while e > tol:
            u_density = delta* l_density / \
                (delta+rho*np.dot(alphas, u_prev)/n)
            e = np.linalg.norm(u_prev-u_density)
            u_prev = u_density
            
        q = np.dot(alphas * payoffs,u_density)
        P = alphas * u_density + np.identity(n)*((n/theta)+np.dot(alphas,u_density))
        v = np.dot(np.linalg.inv(P),q)
        
        new_alphas = np.zeros([n,n])
        for i in range(n):
            for j in range(n):
                if payoffs[i,j] >= v[i]+v[j]:
                    new_alphas[i,j]=1
        
        # Print the number of changes
        print(n**2-(new_alphas == alphas).sum())
        
        if (new_alphas == alphas).all():
            is_converged = True
            keep_iterating = False
        else:
            alphas = new_alphas
            