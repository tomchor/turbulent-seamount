import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from IPython import embed

def convert_gebco_to_meters(ds_gebco, reference_lat=39.4):
    """Convert GEBCO coordinates from degrees to meters, centered on seamount peak"""
    # Find the peak (maximum elevation) in GEBCO data
    peak_location = ds_gebco.elevation.where(ds_gebco.elevation == ds_gebco.elevation.max(), drop=True)
    peak_lat = float(peak_location.lat.mean())
    peak_lon = float(peak_location.lon.mean())

    # Conversion factors
    lat2meter = 111.3e3  # meters per degree latitude
    lon2meter = lat2meter * np.cos(np.deg2rad(reference_lat))  # meters per degree longitude

    # Convert to meters and center on peak
    ds_converted = ds_gebco.copy()
    ds_converted = ds_converted.assign_coords(
        lon=(ds_gebco.lon - peak_lon) * lon2meter,
        lat=(ds_gebco.lat - peak_lat) * lat2meter
    )
    ds_converted = ds_converted.rename({"lon": "x", "lat": "y"})

    return ds_converted

# Load datasets
ds_prep = xr.open_dataset("balanus-bathymetry-preprocessed.nc")
ds_gebco = xr.open_dataset("GEBCO/balanus-gebco_2024_n39.8_s39.0_w-65.8_e-65.0.nc")
ds_gebco_meters = convert_gebco_to_meters(ds_gebco)

# Create figure with two subplots
fig, axes = plt.subplots(ncols=2, figsize=(12, 5),
                         sharex=True, sharey=True,
                         constrained_layout=True)
fig.suptitle("Bathymetry Comparison: Preprocessed vs GEBCO", fontsize=14, fontweight="bold")

# Define common colormap and limits
vmin = min(float(ds_prep.z.min()), float(ds_gebco_meters.elevation.min()))
vmax = max(float(ds_prep.z.max()), float(ds_gebco_meters.elevation.max()))

# Plot 1: Preprocessed bathymetry
ds_prep.z.plot.contourf(
    ax=axes[0],
    x="x", y="y",
    levels=50,
    cmap="terrain",
    vmin=vmin, vmax=vmax,
    add_colorbar=True,
    cbar_kwargs={"shrink": 0.8}
)

axes[0].set_title("GMRT Bathymetry", fontweight="bold")

# Plot 2: GEBCO bathymetry
ds_gebco_meters.elevation.plot.contourf(
    ax=axes[1],
    x="x", y="y",
    levels=50,
    cmap="terrain",
    vmin=vmin, vmax=vmax,
    add_colorbar=True,
    cbar_kwargs={"shrink": 0.8}
)
axes[1].set_title("GEBCO Bathymetry", fontweight="bold")

# Convert axes to km

for ax in axes:
    ax.set_aspect("equal")