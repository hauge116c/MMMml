#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 08:38:21 2025

@author: emilrasmussen
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.integrate import simps
from mpl_toolkits.mplot3d import Axes3D  
from pathlib import Path

# 0) RNG
np.random.seed(2)
#men_x   = np.random.uniform(1.0, 5.0, 50)
#women_x = np.random.uniform(1.0, 5.0, 50)

# 1) PARAMETERS
p = {
    'beta':     0.5,    # Bargaining power
    'r':        0.05,   # Discount rate
    'delta':    0.08,   # Dissolve / divorce rate
    'lambda_m': 3.0,    # Meeting rate, male
    'lambda_f': 3.0,    # Meeting rate, female
    'c_m':      0.0,    # Utility of staying single, men
    'c_f':      0.0,    # Utility of staying single, female
    'Z_max':    100.0,  # Love shock
    'n_z':      101     # Grid for love shock
}

tol = 1e-8
maxit = 500
# 2) LOAD RAW CSVs 
home = Path.home()
path = home / "Documents" / "GitHub" / "MMMml"

men_x = (
    pd.read_csv(path/"data_men1.csv", sep=";", usecols=[0])
      .iloc[:,0].astype(float).to_numpy()
)

women_x = (
    pd.read_csv(path/"data_women1.csv", sep=";", usecols=[0])
      .iloc[:,0].astype(float).to_numpy()
)

# 3) KDE‐SMOOTHED PDF & CDF 
min_inc = min(men_x.min(), women_x.min())
max_inc = max(men_x.max(), women_x.max())
grid    = np.linspace(min_inc, max_inc, 300)

kde_m = gaussian_kde(men_x)
kde_f = gaussian_kde(women_x)

pdf_m = kde_m(grid)
pdf_m = pdf_m / simps(pdf_m, grid)
pdf_f = kde_f(grid) 
pdf_f = pdf_f / simps(pdf_f, grid) 

cdf_m = np.array([simps(pdf_m[:i+1], grid[:i+1]) for i in range(len(grid))])
cdf_f = np.array([simps(pdf_f[:i+1], grid[:i+1]) for i in range(len(grid))])

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,4), sharex=True)
ax1.plot(grid, pdf_m, lw=2, label='Men PDF')
ax1.plot(grid, pdf_f, lw=2, label='Women PDF')
ax1.set(title="KDE‐Smoothed PDF", xlabel="Hourly wage", ylabel="Density")
ax1.legend(); ax1.grid(alpha=0.3)

ax2.plot(grid, cdf_m, lw=2, label='Men CDF')
ax2.plot(grid, cdf_f, lw=2, label='Women CDF')
ax2.set(title="KDE‐Smoothed CDF", xlabel="Hourly wage", ylabel="Cumulative Probability")
ax2.legend(); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# 4) DISCRETE SUPPORT & WEIGHTS
x_vals, cnt_m = np.unique(men_x,   return_counts=True)
y_vals, cnt_f = np.unique(women_x, return_counts=True)

pdf_md = gaussian_kde(np.repeat(x_vals, cnt_m))(x_vals)
pdf_fd = gaussian_kde(np.repeat(y_vals, cnt_f))(y_vals)
pdf_md = pdf_md / simps(pdf_md, x_vals)
pdf_fd = pdf_fd / simps(pdf_fd, y_vals)

# true grid spacings
dx = np.diff(x_vals).mean()
dy = np.diff(y_vals).mean()
wx = pdf_md * dx
wy = pdf_fd * dy

# 5) TWO‐SIDED SHOCK GRID & CDF
C = np.outer(x_vals, y_vals)
z_grid = np.linspace(-p['Z_max'], p['Z_max'], p['n_z'])
G_cdf  = (z_grid + p['Z_max'])/(2*p['Z_max'])

# 6) INITIALIZE EQUILIBRIUM VARIABLES
n_m = len(x_vals)
n_f = len(y_vals)
u_m = np.ones(n_m)
u_f = np.ones(n_f)
s_m = np.zeros(n_m)
s_f = np.zeros(n_f)

# 7) SOLVE J&R EQUILIBRIUM

for it in range(maxit):
    u_m0 = u_m.copy() 
    u_f0 = u_f.copy()
    s_m0 = s_m.copy()
    s_f0 = s_f.copy()

    thresh = -(C - s_m0[:,None] - s_f0[None,:])
    alpha  = 1.0 - np.interp(thresh.ravel(), z_grid, G_cdf,left=1.0, right=0.0).reshape(C.shape)

    int_f = (alpha * u_f0[None,:] * wy[None,:]).sum(axis=1)
    int_m = (alpha * u_m0[:,None] * wx[:,None]).sum(axis=0)

    u_m = p['delta'] * wx / (p['delta'] + p['lambda_m'] * int_f)
    u_f = p['delta'] * wy / (p['delta'] + p['lambda_f'] * int_m)

    U_f     = u_f0.dot(wy)
    U_m     = u_m0.dot(wx)
    denom_f = 1 + (p['lambda_f']*(1-p['beta']))/(p['r']+p['delta'])*U_m
    denom_m = 1 + (p['lambda_m']*p['beta'])/(p['r']+p['delta'])*U_f


    for i, x in enumerate(x_vals):
        num_m = p['c_m']
        for j, y in enumerate(y_vals):
            st = C[i,j] - s_f0[j]
            dz = z_grid[1:] - z_grid[:-1]
            midz = z_grid[:-1] + dz/2
            mask = midz > -st
            if mask.any():
                inc = (st + midz[mask]) * (G_cdf[1:][mask] - G_cdf[:-1][mask])
                num_m += (p['delta']/(p['r']+p['delta'])) * u_f0[j] * inc.sum()
        s_m[i] = num_m / denom_m

    for j, y in enumerate(y_vals):
        num_f = p['c_f']
        for i, x in enumerate(x_vals):
            st = C[i,j] - s_m0[i]
            dz = z_grid[1:] - z_grid[:-1]
            midz = z_grid[:-1] + dz/2
            mask = midz > -st
            if mask.any():
                inc = (st + midz[mask]) * (G_cdf[1:][mask] - G_cdf[:-1][mask])
                num_f += (p['delta']/(p['r']+p['delta'])) * u_m0[i] * inc.sum()
        s_f[j] = num_f / denom_f

    diff = max(
        np.max(abs(u_m-u_m0)),
        np.max(abs(u_f-u_f0)),
        np.max(abs(s_m-s_m0)),
        np.max(abs(s_f-s_f0))
    )
    if diff < tol:
        print(f"Converged in {it+1} iterations (Δ={diff:.2e})")
        break
else:
    print("Warning: did not converge")

# 7) MATCH PROBABILITY (colored contour lines + colorbar)
P_match = alpha
X, Y    = np.meshgrid(y_vals, x_vals)
levels  = np.linspace(P_match.min(), P_match.max(), 20)

plt.figure(figsize=(6,5))
CS = plt.contour(
    Y, X, P_match,
    levels=levels,
    cmap='viridis',
    linewidths=1.5
)
cbar = plt.colorbar(CS)
cbar.set_label(r'$P_{\rm match}(x,y)$')

plt.title("Match Probability")
plt.xlabel("Male hourly wage")
plt.ylabel("Female hourly wage")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 8) COMBINED CONDITIONAL PDF & CDF OF SINGLES VS. MARRIED
UM = u_m.sum()*dx
MM = pdf_md.sum()*dx - UM
UF = u_f.sum()*dy
MF = pdf_fd.sum()*dy - UF

f_sm = u_m            / UM
f_mm = (pdf_md - u_m) / MM
f_sf = u_f            / UF
f_mf = (pdf_fd - u_f) / MF

# PDF
plt.figure(figsize=(8,5))
plt.plot(x_vals, f_sm, 'b-',  lw=2, label='Single Men')
plt.plot(x_vals, f_mm, 'b--', lw=2, label='Married Men')
plt.plot(y_vals, f_sf, 'g-',  lw=2, label='Single Women')
plt.plot(y_vals, f_mf, 'r--', lw=2, label='Married Women')
plt.title('Conditional PDF of hourly wage')
plt.xlabel('Hourly wage'); plt.ylabel('Density')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

# CDF
cdf_sm = np.cumsum(f_sm * dx)
cdf_mm = np.cumsum(f_mm * dx)
cdf_sf = np.cumsum(f_sf * dy)
cdf_mf = np.cumsum(f_mf * dy)

plt.figure(figsize=(8,5))
plt.plot(x_vals, cdf_sm, 'b-',  lw=2, label='Single Men')
plt.plot(x_vals, cdf_mm, 'b--', lw=2, label='Married Men')
plt.plot(y_vals, cdf_sf, 'g-',  lw=2, label='Single Women')
plt.plot(y_vals, cdf_mf, 'r--', lw=2, label='Married Women')
plt.title('Conditional CDF of hourly wage')
plt.xlabel('Hourly wage'); plt.ylabel('Cumulative Probability')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

# 9a) Joint PDF of Married Couples
joint_raw = np.outer(u_m, u_f) * alpha
joint_pdf = joint_raw / (joint_raw.sum()*dx*dy)

Xj, Yj = np.meshgrid(y_vals, x_vals)
fig1   = plt.figure(figsize=(6,5))
ax1    = fig1.add_subplot(111, projection='3d')
surf1  = ax1.plot_surface(
    Xj, Yj, joint_pdf,
    cmap='viridis', edgecolor='none'
)
ax1.set_title('Joint PDF of Married Couples')
ax1.set_xlabel('Female hourly wage')
ax1.set_ylabel('Male hourly wage')
ax1.set_zlabel('Density')
fig1.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
plt.tight_layout()
plt.show()


# 9b) Threshold Surface: C(x,y) – s_m – s_f
threshold = C - s_m[:,None] - s_f[None,:]

fig2   = plt.figure(figsize=(6,5))
ax2    = fig2.add_subplot(111, projection='3d')
surf2  = ax2.plot_surface(
    Xj, Yj, threshold,
    cmap='coolwarm', edgecolor='none'
)
ax2.set_title('Threshold: C(x,y) - s_m - s_f')
ax2.set_xlabel('Female hourly wage')
ax2.set_ylabel('Male hourly wage')
ax2.set_zlabel('Threshold')
fig2.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
plt.tight_layout()
plt.show()

P_match = alpha
X, Y = np.meshgrid(x_vals, y_vals)
plt.figure(figsize=(6,5))
pcm = plt.pcolormesh(
    X, Y, P_match.T, 
    cmap='viridis', shading='auto'
)
plt.xlabel("Male hourly wage"); plt.ylabel("Female hourly wage")
plt.title("Endogenous Match Probability")
plt.colorbar(pcm, label=r'$P_{\rm match}(x,y)$')
plt.tight_layout(); plt.show()
