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

def low_pass_filter(da, filter_scale):
    """
    Apply a low-pass filter to 2D elevation data using xarray's rolling average.
    Assumes periodic boundary conditions in both directions.
    
    Parameters:
    -----------
    da : xarray.DataArray
        2D DataArray of elevation data with x and y coordinates
    filter_scale : float
        Scale of the filter in the same units as the coordinates
        (e.g., meters if coordinates are in meters)
    
    Returns:
    --------
    xarray.DataArray
        High-pass filtered elevation data with the same dimensions and coordinates
    """
    # Calculate window sizes in grid points
    dx = float(da.x.diff('x').mean())
    dy = float(da.y.diff('y').mean())
    
    # Convert filter scale to grid points and round to nearest odd number
    window_x = int(2 * (filter_scale / dx) // 2) + 1
    window_y = int(2 * (filter_scale / dy) // 2) + 1
    
    # Ensure minimum window size of 3
    window_x = max(3, window_x)
    window_y = max(3, window_y)
    
    # Apply rolling mean (low-pass) with periodic boundary conditions
    # Use min_periods=1 to handle edge cases
    low_pass = da.pad(dict(x=window_x, y=window_y), mode="wrap").rolling(
        x=window_x,
        y=window_y,
        center=True,
        min_periods=1
    ).mean()
    
    # Add information about the filtering
    low_pass.attrs["filter_scale"] = float(filter_scale)
    low_pass.attrs["filter_type"] = "low-pass (periodic)"
    low_pass.attrs["window_size_x"] = window_x
    low_pass.attrs["window_size_y"] = window_y
    
    return low_pass

def high_pass_filter(da, filter_scale):
    high_pass = da - low_pass_filter(da, filter_scale)

    # Add information about the filtering
    high_pass.attrs["filter_scale"] = float(filter_scale)
    high_pass.attrs["filter_type"] = "high-pass (periodic)"
    high_pass.attrs["window_size_x"] = window_x
    high_pass.attrs["window_size_y"] = window_y
    
    return high_pass
#---

ds = xr.open_dataset("GMRT/GMRTv4_3_1_20250502topo.nc")
ds = ds.dropna("lat", how="all").dropna("lon", how="all")

ds["detrended_elevation"] = detrend_elevation(ds.z)

maximum_point = get_max_location_argmax(ds.detrended_elevation)
ds = ds.assign_coords(lat = ds.lat - maximum_point["lat"], lon = ds.lon - maximum_point["lon"])

# The resolution of the GEBCO 2024 datasets is 15 arcseconds. At 40°-ish latitude then
Δlat = ds.lat.diff("lat").mean().item()
Δlon = ds.lon.diff("lon").mean().item()

# Or better yet
lat2meter = 111.3e3 # meters
lon2meter = lat2meter * np.cos(np.deg2rad(maximum_point["lat"]))

degrees_to_arcseconds = 60 * 60
print(f"Dataset has spacing of {Δlat * degrees_to_arcseconds:.2f} arcseconds in latitude and {Δlon *
      degrees_to_arcseconds:.2f} arcseconds in longitude")

ds = ds.assign_coords(lat = ds.lat * lat2meter, lon = ds.lon * lon2meter)
ds = ds.rename(lon="x", lat="y")

coarse_elevation = ds.z.coarsen(x=10, y=10, boundary="trim").mean()
trend = low_pass_filter(coarse_elevation, filter_scale=40e3)
pause

ds["trend_elevation"] = low_pass_filter(ds.detrended_elevation, filter_scale=40e3)
ds["detrended_elevation2"] = ds.detrended_elevation - ds.trend_elevation
pause

ds.detrended_elevation.isel(x=-1).plot()
ds.detrended_elevation2.isel(x=-1).plot()
pause


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
    plt.gca().set_aspect('equal')

if True:
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_box_aspect((ds.x[-1] - ds.x[0], ds.y[-1] - ds.y[0], 5*ds.H))  # aspect ratio is 1:1:1 in data space
    X = (ds.x + 0 * ds.y).T
    Y = (ds.y + 0 * ds.x)
    surf = ax.plot_surface(X, Y, ds.detrended_elevation, cmap=plt.cm.viridis, linewidth=0, antialiased=False)

encoding = { var : dict(zlib=True, complevel=9, shuffle=True) for var in ds.data_vars }
ds.to_netcdf("balanus-bathymetry-preprocessed.nc", encoding = encoding)
