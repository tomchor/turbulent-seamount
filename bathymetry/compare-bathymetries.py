import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from IPython import embed

def convert_GEBCO_to_meters(ds_GEBCO, reference_lat=39.4):
    """Convert GEBCO coordinates from degrees to meters, centered on seamount peak"""
    # Find the peak (maximum elevation) in GEBCO data
    peak_location = ds_GEBCO.elevation.where(ds_GEBCO.elevation == ds_GEBCO.elevation.max(), drop=True)
    peak_lat = float(peak_location.lat.mean())
    peak_lon = float(peak_location.lon.mean())

    # Conversion factors
    lat2meter = 111.3e3  # meters per degree latitude
    lon2meter = lat2meter * np.cos(np.deg2rad(reference_lat))  # meters per degree longitude

    # Convert to meters and center on peak
    ds_converted = ds_GEBCO.copy()
    ds_converted = ds_converted.assign_coords(
        lon=(ds_GEBCO.lon - peak_lon) * lon2meter,
        lat=(ds_GEBCO.lat - peak_lat) * lat2meter
    )
    ds_converted = ds_converted.rename({"lon": "x", "lat": "y"})

    return ds_converted

# Load datasets
ds_GMRT = xr.open_dataset("balanus-GMRT-bathymetry-preprocessed.nc")
ds_GEBCO = xr.open_dataset("GEBCO/balanus-gebco_2024_n39.8_s39.0_w-65.8_e-65.0.nc")
ds_GEBCO = convert_GEBCO_to_meters(ds_GEBCO)

ds_GMRT = ds_GMRT.sel(x=slice(-23000, 23000), y=slice(-22000, 22000))
ds_GEBCO = ds_GEBCO.sel(x=slice(-23000, 23000), y=slice(-22000, 22000))

# Calculate gradients using xarray
ds_GMRT_grad_mag = np.sqrt(ds_GMRT.z.differentiate('x')**2 + ds_GMRT.z.differentiate('y')**2)
ds_GEBCO_grad_mag = np.sqrt(ds_GEBCO.elevation.differentiate('x')**2 + ds_GEBCO.elevation.differentiate('y')**2)

# Create figure with 2x2 subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10),
                         sharex=True, sharey=True,
                         constrained_layout=True)
fig.suptitle("Bathymetry Comparison: GMRT vs GEBCO", fontsize=14, fontweight="bold")

# Define common colormap and limits
vmin = min(float(ds_GMRT.z.min()), float(ds_GEBCO.elevation.min()))
vmax = max(float(ds_GMRT.z.max()), float(ds_GEBCO.elevation.max()))

# Define common gradient limits
grad_vmin = 0
grad_vmax = 1

# Calculate max heights for display
gmrt_max_height = float(ds_GMRT.z.max())
gebco_max_height = float(ds_GEBCO.elevation.max())

# Calculate mean spacing in x and y directions
gmrt_dx = float(ds_GMRT.x.diff('x').mean())
gmrt_dy = float(ds_GMRT.y.diff('y').mean())
gebco_dx = float(ds_GEBCO.x.diff('x').mean())
gebco_dy = float(ds_GEBCO.y.diff('y').mean())

# Calculate bathymetry volume above minimum elevation
# Volume = integrated elevation above the minimum elevation
gmrt_min_elev = float(ds_GMRT.z.min())
gebco_min_elev = gmrt_min_elev

# Calculate volume by integrating elevation above minimum
gmrt_volume = float((ds_GMRT.z - gmrt_min_elev).integrate(['x', 'y'])) / 1e9  # Convert to km³
gebco_volume = float((ds_GEBCO.elevation - gebco_min_elev).integrate(['x', 'y'])) / 1e9  # Convert to km³

# Plot 1: GMRT bathymetry
im1 = ds_GMRT.z.plot.imshow(
    ax=axes[0,0],
    x="x", y="y",
    cmap="terrain",
    vmin=vmin, vmax=vmax,
    add_colorbar=False
)
axes[0,0].set_title(f"GMRT Bathymetry\nMax height: {gmrt_max_height:.0f} m | Volume: {gmrt_volume:.2f} km³\nSpacing: Δx={gmrt_dx:.0f} m, Δy={gmrt_dy:.0f} m", fontweight="bold")

# Plot 2: GEBCO bathymetry
im2 = ds_GEBCO.elevation.plot.imshow(
    ax=axes[0,1],
    x="x", y="y",
    cmap="terrain",
    vmin=vmin, vmax=vmax,
    add_colorbar=False
)
axes[0,1].set_title(f"GEBCO Bathymetry\nMax height: {gebco_max_height:.0f} m | Volume: {gebco_volume:.2f} km³\nSpacing: Δx={gebco_dx:.0f} m, Δy={gebco_dy:.0f} m", fontweight="bold")

# Add shared colorbar for bathymetry (top row)
cbar1 = fig.colorbar(im1, ax=axes[0,:], shrink=0.8, aspect=30)
cbar1.set_label("Elevation (m)", rotation=270, labelpad=20)

# Plot 3: GMRT bathymetry gradient
im3 = ds_GMRT_grad_mag.plot.imshow(
    ax=axes[1,0],
    x="x", y="y",
    cmap="plasma",
    vmin=grad_vmin, vmax=grad_vmax,
    add_colorbar=False
)
axes[1,0].set_title("GMRT Gradient Magnitude", fontweight="bold")

# Plot 4: GEBCO bathymetry gradient
im4 = ds_GEBCO_grad_mag.plot.imshow(
    ax=axes[1,1],
    x="x", y="y",
    cmap="plasma",
    vmin=grad_vmin, vmax=grad_vmax,
    add_colorbar=False
)
axes[1,1].set_title("GEBCO Gradient Magnitude", fontweight="bold")

# Add shared colorbar for gradients (bottom row)
cbar2 = fig.colorbar(im3, ax=axes[1,:], shrink=0.8, aspect=30)
cbar2.set_label("Gradient Magnitude", rotation=270, labelpad=20)

# Convert axes to km and set equal aspect
for i in range(2):
    for j in range(2):
        ax = axes[i,j]
        ax.set_aspect("equal")