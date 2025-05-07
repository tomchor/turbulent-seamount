import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from cmocean import cm

#+++ Utility functions
def get_max_location_argmax(da):
    """Find the location of the maximum using argmax"""
    # Get the indices of the maximum value
    flat_idx = da.argmax().item()

    # Convert flat index to multi-dimensional indices
    idx_coords = np.unravel_index(flat_idx, da.shape)

    # Create a dictionary mapping dimension names to their coordinate values
    result = {}
    for dim_idx, dim_name in enumerate(da.dims):
        result[dim_name] = da[dim_name].values[idx_coords[dim_idx]]

    # Add the maximum value itself
    result['value'] = da.max().item()

    return result

from scipy import linalg
def detrend_elevation(da):
    """
    Remove a linear trend (slope) from 2D elevation data.

    Parameters:
    -----------
    da : xarray.DataArray
        2D DataArray of elevation data with lon and lat coordinates

    Returns:
    --------
    xarray.DataArray
        Detrended elevation data with the same dimensions and coordinates
    """
    # Extract coordinates and values
    x_coords = da.lon.values
    y_coords = da.lat.values
    elevation = da.values

    # Check if we're working with 1D coordinate arrays
    if x_coords.ndim == 1 and y_coords.ndim == 1:
        # Create meshgrid for 1D coordinates
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    else:
        # Assume coordinates are already in grid form
        x_grid, y_grid = x_coords, y_coords

    # Flatten the arrays for fitting
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    z_flat = elevation.flatten()

    # Remove NaN values for fitting
    valid_idx = ~np.isnan(z_flat)
    x_valid = x_flat[valid_idx]
    y_valid = y_flat[valid_idx]
    z_valid = z_flat[valid_idx]

    # Create design matrix for plane fitting: z = a*lon + b*lat + c
    A = np.column_stack((x_valid, y_valid, np.ones_like(x_valid)))

    # Solve for coefficients using least squares
    coeffs, residuals, rank, s = linalg.lstsq(A, z_valid)

    # Extract coefficients
    a, b, c = coeffs

    # Calculate the trend plane across the entire grid
    trend = a * x_grid + b * y_grid + c

    # Subtract the trend from the original data
    detrended = elevation - trend
    detrended = detrended - detrended.min()

    # Create a new DataArray with the detrended data
    detrended_da = xr.DataArray(
        detrended,
        coords=da.coords,
        dims=da.dims,
        attrs=da.attrs
    )

    # Add information about the removed trend
    detrended_da.attrs["detrending_info"] = f"Removed plane: z = {a:.6f}*lon + {b:.6f}*lat + {c:.6f}"
    detrended_da.attrs["slope_x"] = float(a)
    detrended_da.attrs["slope_y"] = float(b)
    detrended_da.attrs["intercept"] = float(c)

    return detrended_da
#---

ds1 = xr.load_dataset("GEBCO/balanus-gebco_2024_n39.8_s39.0_w-65.8_e-65.0.nc")
ds2 = xr.load_dataset("GMRT/GMRTv4_3_1_20250502topo.nc")

if False:
    fig, axes = plt.subplots(ncols=2, figsize=(12, 6), sharex=True, sharey=True, layout='constrained')
    
    # Find global min and max across all datasets
    vmin = min(ds1.elevation.min(), ds2.z.min())
    vmax = max(ds1.elevation.max(), ds2.z.max())
    
    ds1.elevation.plot(ax=axes[0], add_colorbar=False, vmin=vmin, vmax=vmax)
    im = ds2.z.plot(ax=axes[1], vmin=vmin, vmax=vmax)

ds = ds2
ds["detrended_elevation"] = detrend_elevation(ds.elevation)

maximum_point = get_max_location_argmax(ds.detrended_elevation)
ds = ds.assign_coords(lat = ds.lat - maximum_point["lat"], lon = ds.lon - maximum_point["lon"])

# The resolution of the GEBCO 2024 datasets is 15 arcseconds. At 40°-ish latitude then
Δlat = 463.8 # meters
Δlon = 355.3 # meters

# Or better yet
lat2meter = 111_130 # meters
lon2meter = 85_390 # meters

ds = ds.assign_coords(lat = ds.lat * lat2meter, lon = ds.lon * lon2meter)
ds = ds.rename(lon="x", lat="y")

# Useful to estimate full width at half maximum (FWHM)
ds["half_maximum_ring"] = ds.detrended_elevation.where(abs(ds.detrended_elevation - ds.detrended_elevation.max()/2) < 100)
ds["distance_from_peak"] = np.sqrt(ds.x**2 + ds.y**2)

ds.attrs["H"] = maximum_point["value"]
ds.attrs["FWHM"] = ds.distance_from_peak.where(np.logical_not(np.isnan(ds.half_maximum_ring))).mean().item()
ds.attrs["δ"] = ds.H / ds.FWHM

if True:
    plt.figure()
    ds.distance_from_peak.plot.contourf(levels=[0, 5e3, 10e3, 15e3, 20e3])
    ds.half_maximum_ring.plot.contour(colors="k")

    plt.gca().set_title(f"Full width at half maximum is {ds.FWHM:.2f} m")

if True:
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_box_aspect((ds.x[-1] - ds.x[0], ds.y[-1] - ds.y[0], 5*ds.H))  # aspect ratio is 1:1:1 in data space
    X = ds.x + 0 * ds.y
    Y = (ds.y + 0 * ds.x).T
    surf = ax.plot_surface(X, Y, ds.detrended_elevation, cmap=plt.cm.viridis, linewidth=0, antialiased=False)

encoding = { var : dict(zlib=True, complevel=9, shuffle=True) for var in ds.data_vars }
ds.to_netcdf("balanus-bathymetry-preprocessed.nc", encoding = encoding)
