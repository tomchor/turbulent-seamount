#!/usr/bin/env python3
"""
Plot histograms of interesting seamount quantities
from the seamount_data.nc dataset.

This script creates a multi-panel figure showing histograms of:
- Froude number (height-based)
- Rossby number
- Slope Burger number (Ro/Fr)
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Set up matplotlib for better plots
plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["font.size"] = 10


# Load the seamount data
print("Loading seamount data...")
ds = xr.open_dataset("seamount_data.nc")
ds["slope_burger_height"] = ds["rossby_number"] / ds["froude_height"]
print(f"Loaded data for {ds.dims["index"]} seamounts")

# Variables to plot
variables = ["froude_height", "rossby_number", "slope_burger_height"]
var_names = ["Froude Number", "Rossby Number", "Slope Burger Number"]
colors = ["red", "blue", "green"]

# Determine common x-axis range for all variables (in log space)
all_data = []
for var in variables:
    clean_data = ds[var].values[np.isfinite(ds[var].values)]
    positive_data = clean_data[clean_data > 0]
    if len(positive_data) > 0:
        all_data.extend(np.log10(positive_data))

x_min, x_max = -3, 2  # Default range

# Create figure with two rows of subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10), sharex=True, sharey="row")

def plot_histogram_row(axes_row, dataset, variables, var_names, colors, title_suffix):
    """Plot a row of histograms for the given dataset and variables"""
    bins = np.linspace(x_min, x_max, 50)

    for i, (var, name, color) in enumerate(zip(variables, var_names, colors)):
        ax = axes_row[i]

        # Get clean data and convert to log10
        clean_data = dataset[var].values[np.isfinite(dataset[var].values)]
        positive_data = clean_data[clean_data > 0]

        if len(positive_data) > 0:
            log_data = np.log10(positive_data)

            # Create xarray DataArray for plotting
            log_da = xr.DataArray(log_data, dims=["seamount"])

            # Plot histogram
            log_da.plot.hist(ax=ax, bins=bins, density=False, alpha=0.7,
                            color=color, edgecolor="black", linewidth=0.5)

            # Add statistics
            median_val = np.median(positive_data)
            mean_val = np.mean(positive_data)
            n_val = len(positive_data)

            stats_text = f"N = {n_val:,}\nMedian = {median_val:.2e}\nMean = {mean_val:.2e}"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                    verticalalignment="top", fontsize=8,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        else:
            # Handle case where no data exists
            ax.text(0.5, 0.5, f"No data for {name}\n{title_suffix}",
                    transform=ax.transAxes, ha="center", va="center",
                    bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8))

        # Set common formatting
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel(f"log₁₀({name})")
        ax.set_ylabel("Count")
        ax.set_title(f"Histogram of {name} ({title_suffix})")
        ax.grid(True, alpha=0.3)

# Filter data for high southern latitudes (< -40°)
ds_low_lat = ds.where(ds.latitude < -40, drop=True)

# Plot both rows using the helper function
plot_histogram_row(axes[0], ds, variables, var_names, colors, "All Data")
plot_histogram_row(axes[1], ds_low_lat, variables, var_names, colors, "Latitude < -40°")

plt.tight_layout()