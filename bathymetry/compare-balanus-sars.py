import numpy as np
import xarray as xr
from scipy import interpolate

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

def interpolate_2d_scipy(da, method='linear'):
    """Interpolate NaN values in a 2D xarray DataArray using scipy."""
    # Get coordinates and data values
    y_coords, x_coords = np.meshgrid(da.y.values, da.x.values, indexing='ij')

    # Find valid data points
    valid_mask = ~np.isnan(da.values)
    points = np.column_stack((y_coords[valid_mask], x_coords[valid_mask]))
    values = da.values[valid_mask]

    # Define target grid points (all points)
    grid_y, grid_x = np.meshgrid(da.y.values, da.x.values, indexing='ij')

    # Perform interpolation
    filled_data = interpolate.griddata(
        points, values, (grid_y, grid_x),
        method=method, fill_value=np.nan
    )

    # Create new DataArray with interpolated values
    return xr.DataArray(
        filled_data,
        dims=da.dims,
        coords=da.coords
    )
#---

#+++ Open and detrend elevation
sars = xr.open_dataset("GMRT/DEM_20251018DEM_sars.nc")
bala = xr.open_dataset("GMRT/GMRTv4_3_1_20250502topo_balanus.nc")

sars = sars.assign_coords(lon = sars.lon - sars.lon.mean(), lat = sars.lat - sars.lat.mean())
bala = bala.assign_coords(lon = bala.lon - bala.lon.mean(), lat = bala.lat - bala.lat.mean())

sars.sel(lat=0, method="nearest").z.plot.scatter(x="lon", edgecolor="none", label="sars")
bala.sel(lat=0, method="nearest").z.plot.scatter(x="lon", edgecolor="none", label="balanus")

pause
ds = ds.dropna("lat", how="all").dropna("lon", how="all")

ds["detrended_elevation"] = detrend_elevation(ds.z)

maximum_point = get_max_location_argmax(ds.detrended_elevation)
ds = ds.assign_coords(lat = ds.lat - maximum_point["lat"], lon = ds.lon - maximum_point["lon"])
ds.attrs["H"] = maximum_point["value"]
#---

#+++ Convert from lat lon to meters
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
#---

#+++ Estimate full width at half maximum (FWHM)
area_at_HM = xr.ones_like(ds.detrended_elevation).where(ds.detrended_elevation > ds.H/2, other=0).integrate(("x", "y"))
ds.attrs["FWHM"] = float(2 * np.sqrt(area_at_HM / np.pi))
ds["distance_from_peak"] = np.sqrt(ds.x**2 + ds.y**2)
ds.attrs["δ"] = float(ds.H / ds.attrs["FWHM"])
#---

#+++ Make it periodic
ringed_periodic_elevation = ds.detrended_elevation.where(ds.distance_from_peak < 1*ds.FWHM).where(ds.distance_from_peak < 1.2*ds.FWHM, other=0)
pause
ds["periodic_elevation"] = interpolate_2d_scipy(ringed_periodic_elevation)

ds = ds.drop_vars(["detrended_elevation", "distance_from_peak"])

# Coarsen to reduce points by half in x and y directions
ds = ds.coarsen(x=2, y=2, boundary="pad").mean()
#---

#+++ Extend the dataset in x and y directions using native xarray functions
# Calculate extension parameters
dx = ds.x.diff("x").mean().item()
dy = ds.y.diff("y").mean().item()

x_min, x_max = ds.x.min().item(), ds.x.max().item()
y_min, y_max = ds.y.min().item(), ds.y.max().item()

# Create extension coordinates by progressive concatenation
# Calculate target extents
x_target_min = 1.5 * x_min
x_target_max = 1.5 * x_max
y_target_min = 1.5 * y_min
y_target_max = 1.5 * y_max

# Create western extension points by going backwards from x_min
x_west = []
x_current = x_min - dx
while x_current >= x_target_min:
    x_west.insert(0, x_current)  # Insert at beginning to maintain order
    x_current -= dx

# Create eastern extension points by going forwards from x_max
x_east = []
x_current = x_max + dx
while x_current <= x_target_max:
    x_east.append(x_current)
    x_current += dx

# Create southern extension points by going backwards from y_min
y_south = []
y_current = y_min - dy
while y_current >= y_target_min:
    y_south.insert(0, y_current)  # Insert at beginning to maintain order
    y_current -= dy

# Create northern extension points by going forwards from y_max
y_north = []
y_current = y_max + dy
while y_current <= y_target_max:
    y_north.append(y_current)
    y_current += dy

# Concatenate all coordinates
x_extended = np.concatenate([x_west, ds.x.values, x_east])
y_extended = np.concatenate([y_south, ds.y.values, y_north])

ds_extended = ds.reindex(x=x_extended, y=y_extended, fill_value=np.nan)

ds_extended["periodic_elevation"] = ds_extended.periodic_elevation.fillna(0)
#---

encoding = { var : dict(zlib=True, complevel=9, shuffle=True) for var in ds_extended.data_vars }
ds_extended.to_netcdf("balanus-GMRT-bathymetry-preprocessed.nc", encoding = encoding)
