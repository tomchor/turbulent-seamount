import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import xrft
from scipy import ndimage

def spectral_cutoff_filter(data, cutoff_wavelength, x_coord, y_coord):
    """
    Apply a sharp spectral cut-off filter to 2D data using xrft.

    Parameters:
    -----------
    data : xarray.DataArray
        2D DataArray to be filtered
    cutoff_wavelength : float
        Cutoff wavelength in same units as coordinates
    x_coord, y_coord : str
        Names of x and y coordinate dimensions

    Returns:
    --------
    filtered_data : xarray.DataArray
        Filtered 2D DataArray
    """
    # 2D FFT using xrft
    data_fft = xrft.fft(data, dim=[y_coord, x_coord], real_dim=x_coord)
    freq_x = data_fft.coords[f'freq_{x_coord}']
    freq_y = data_fft.coords[f'freq_{y_coord}']

    # 2D frequency grids
    FREQ_X, FREQ_Y = xr.broadcast(freq_x, freq_y)

    # Wavenumber magnitude (frequency = wavenumber / 2π)
    K = 2 * np.pi * np.sqrt(FREQ_X**2 + FREQ_Y**2)

    # Cutoff wavenumber (2π/wavelength)
    k_cutoff = 2 * np.pi / cutoff_wavelength
    filter_mask = K <= k_cutoff
    filtered_fft = data_fft * filter_mask

    # Transform back to spatial domain
    filtered_data = xrft.ifft(filtered_fft, dim=[f'freq_{y_coord}', f'freq_{x_coord}'],
                              real_dim=f'freq_{x_coord}')

    return filtered_data.real


def gaussian_filter_2d(data, sigma_x, sigma_y):
    """
    Apply Gaussian filtering to 2D data using scipy.ndimage.

    Parameters:
    -----------
    data : xarray.DataArray
        2D DataArray to be filtered
    sigma_x, sigma_y : float
        Standard deviation for Gaussian kernel in x and y directions (in grid points)

    Returns:
    --------
    filtered_data : xarray.DataArray
        Filtered 2D DataArray
    """
    # Apply Gaussian filter to the data values
    filtered_values = ndimage.gaussian_filter(data.values, sigma=[sigma_y, sigma_x])

    # Create new DataArray with filtered values but same coordinates and attributes
    filtered_data = data.copy()
    filtered_data.values = filtered_values

    return filtered_data


# Read the bathymetry data
print("Reading bathymetry data...")
ds = xr.open_dataset("balanus-bathymetry-preprocessed.nc")

# Extract variables as DataArrays (keep xarray structure)
elevation_da = ds["periodic_elevation"]

# Define multiple cutoff wavelengths for comparison
cutoff_wavelengths = ds.FWHM  * np.array([0.05, 0.1, 0.2, 0.4, 0.8])  # meters
print(f"\nApplying spectral filters with cutoff wavelengths: {[w/1000 for w in cutoff_wavelengths]} km")

# Get grid spacing for Gaussian filtering
dx = float(elevation_da.x.diff('x').mean().values)
dy = float(elevation_da.y.diff('y').mean().values)

# Apply spectral filters for all wavelengths
spectral_filtered_data = []
for cutoff_wavelength in cutoff_wavelengths:
    print(f"Processing spectral filter with cutoff wavelength: {cutoff_wavelength/1000:.1f} km")
    filtered_elevation_da = spectral_cutoff_filter(elevation_da, cutoff_wavelength, "x", "y")
    spectral_filtered_data.append(filtered_elevation_da)

# Apply Gaussian filters for corresponding length scales
gaussian_filtered_data = []
for cutoff_wavelength in cutoff_wavelengths:
    # Convert wavelength to sigma in grid points
    # For Gaussian filter, sigma ≈ wavelength / (2π * grid_spacing)
    sigma_x = cutoff_wavelength / (2 * np.pi * dx)
    sigma_y = cutoff_wavelength / (2 * np.pi * dy)

    print(f"Processing Gaussian filter with sigma: {sigma_x:.1f} grid points ({cutoff_wavelength/1000:.1f} km scale)")
    gaussian_filtered_da = gaussian_filter_2d(elevation_da, sigma_x, sigma_y)
    gaussian_filtered_data.append(gaussian_filtered_da)

print("All filtering completed!")



# Create comparison figure: Gaussian vs Spectral filtering
fig_comparison = plt.figure(figsize=(25, 12))

# Convert coordinates to km for better display
elevation_da_km = elevation_da.assign_coords(x=elevation_da.x/1000, y=elevation_da.y/1000)
gaussian_filtered_data_km = [filtered_da.assign_coords(x=filtered_da.x/1000, y=filtered_da.y/1000)
                            for filtered_da in gaussian_filtered_data]
spectral_filtered_data_km = [filtered_da.assign_coords(x=filtered_da.x/1000, y=filtered_da.y/1000)
                            for filtered_da in spectral_filtered_data]

# Common colormap settings for comparison
all_data = [elevation_da] + gaussian_filtered_data + spectral_filtered_data
vmin = min([data.min().values for data in all_data])
vmax = max([data.max().values for data in all_data])

# Top row: Gaussian filtering
for i, (cutoff_wavelength, gaussian_da_km) in enumerate(zip(cutoff_wavelengths, gaussian_filtered_data_km)):
    ax = fig_comparison.add_subplot(2, 5, i+1)  # positions 1-5 (top row)
    im = gaussian_da_km.plot.contourf(ax=ax, x='x', y='y', levels=50, cmap='terrain',
                                     vmin=vmin, vmax=vmax, extend='both', add_colorbar=False)
    ax.set_title(f'Gaussian (σ={cutoff_wavelength/1000:.2f} km)', fontsize=11)
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_aspect('equal')

# Bottom row: Spectral filtering
for i, (cutoff_wavelength, spectral_da_km) in enumerate(zip(cutoff_wavelengths, spectral_filtered_data_km)):
    ax = fig_comparison.add_subplot(2, 5, i+6)  # positions 6-10 (bottom row)
    im = spectral_da_km.plot.contourf(ax=ax, x='x', y='y', levels=50, cmap='terrain',
                                     vmin=vmin, vmax=vmax, extend='both', add_colorbar=False)
    ax.set_title(f'Spectral (λ>{cutoff_wavelength/1000:.2f} km)', fontsize=11)
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_aspect('equal')

# Add row labels
fig_comparison.text(0.02, 0.75, 'Gaussian\nFiltering', fontsize=14, fontweight='bold',
                   rotation=90, va='center', ha='center')
fig_comparison.text(0.02, 0.25, 'Spectral\nFiltering', fontsize=14, fontweight='bold',
                   rotation=90, va='center', ha='center')

# Add colorbar for comparison figure
plt.tight_layout()
cbar_comparison = fig_comparison.colorbar(im, ax=fig_comparison.axes, shrink=0.6, aspect=40, pad=0.02)
cbar_comparison.set_label('Elevation (m)')

# Add overall title for comparison
fig_comparison.suptitle('Filtering Comparison: Gaussian vs Spectral Filtering at Different Length Scales',
                       fontsize=16, y=0.95)

plt.show()

# Display statistics comparison
print(f"\nStatistics Comparison:")
print(f"Original elevation - min: {elevation_da.min().values:.1f} m, max: {elevation_da.max().values:.1f} m, std: {elevation_da.std().values:.1f} m")
print("\nGaussian Filtering:")
for i, (cutoff_wavelength, filtered_da) in enumerate(zip(cutoff_wavelengths, gaussian_filtered_data)):
    print(f"  Scale {cutoff_wavelength/1000:.2f} km - min: {filtered_da.min().values:.1f} m, max: {filtered_da.max().values:.1f} m, std: {filtered_da.std().values:.1f} m")
print("\nSpectral Filtering:")
for i, (cutoff_wavelength, filtered_da) in enumerate(zip(cutoff_wavelengths, spectral_filtered_data)):
    print(f"  Cutoff {cutoff_wavelength/1000:.2f} km - min: {filtered_da.min().values:.1f} m, max: {filtered_da.max().values:.1f} m, std: {filtered_da.std().values:.1f} m")

print("\nScript completed successfully!")