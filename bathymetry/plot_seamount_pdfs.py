#!/usr/bin/env python3
"""
Plot probability density functions (PDFs) of interesting seamount quantities
from the seamount_data.nc dataset.

This script creates a multi-panel figure showing PDFs of:
- Froude number (height-based)
- Rossby number  
- Burger number (height-based)
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Set up matplotlib for better plots
plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["font.size"] = 10


# Load the seamount data
print("Loading seamount data...")
ds = xr.open_dataset('seamount_data.nc')
ds["slope_burger_height"] = ds["rossby_number"] / ds["froude_height"]
print(f"Loaded data for {ds.dims['index']} seamounts")

# Variables to plot
variables = ['froude_height', 'rossby_number', 'slope_burger_height']
var_names = ['Froude Number', 'Rossby Number', 'Slope Burger Number']
colors = ['red', 'blue', 'green']

# Determine common x-axis range for all variables (in log space)
all_data = []
for var in variables:
    clean_data = ds[var].values[np.isfinite(ds[var].values)]
    positive_data = clean_data[clean_data > 0]
    if len(positive_data) > 0:
        all_data.extend(np.log10(positive_data))

x_min, x_max = -3, 2  # Default range

# Create figure with subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

# Plot histograms for each variable
for i, (var, name, color) in enumerate(zip(variables, var_names, colors)):
    ax = axes.flat[i]
    
    # Get clean data and convert to log10
    clean_data = ds[var].values[np.isfinite(ds[var].values)]
    positive_data = clean_data[clean_data > 0]
    
    if len(positive_data) > 0:
        log_data = np.log10(positive_data)
        
        # Create xarray DataArray for plotting
        log_da = xr.DataArray(log_data, dims=['seamount'])
        
        # Plot histogram (which approximates PDF)
        log_da.plot.hist(ax=ax, bins=50, density=True, alpha=0.7, 
                        color=color, edgecolor='black', linewidth=0.5)
        
        # Set common x-axis range
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel(f"log₁₀({name})")
        ax.set_ylabel("Probability Density")
        ax.set_title(f"PDF of {name}")
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        median_val = np.median(positive_data)
        mean_val = np.mean(positive_data)
        n_val = len(positive_data)
        
        stats_text = f"N = {n_val:,}\nMedian = {median_val:.2e}\nMean = {mean_val:.2e}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Use the fourth panel for a combined comparison plot
ax_combined = axes[1, 1]

for var, name, color in zip(variables, var_names, colors):
    clean_data = ds[var].values[np.isfinite(ds[var].values)]
    positive_data = clean_data[clean_data > 0]
    
    if len(positive_data) > 0:
        log_data = np.log10(positive_data)
        log_da = xr.DataArray(log_data, dims=['seamount'])
        
        # Plot histogram with transparency for overlay
        log_da.plot.hist(ax=ax_combined, bins=50, density=True, alpha=0.6,
                        color=color, label=f'{name.split()[0]} (N={len(positive_data):,})',
                        histtype='stepfilled')

ax_combined.set_xlim(x_min, x_max)
ax_combined.set_xlabel("log₁₀(Parameter Value)")
ax_combined.set_ylabel("Probability Density")
ax_combined.set_title("Comparison of Parameter Distributions")
ax_combined.grid(True, alpha=0.3)
ax_combined.legend()

plt.tight_layout()