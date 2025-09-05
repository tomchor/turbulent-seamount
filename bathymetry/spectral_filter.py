import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import xrft

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


# Read the bathymetry data
print("Reading bathymetry data...")
ds = xr.open_dataset("balanus-bathymetry-preprocessed.nc")

# Extract variables as DataArrays (keep xarray structure)
elevation_da = ds["periodic_elevation"]

# Define cutoff wavelength (you can adjust this)
# For example, filter out wavelengths shorter than 10 km
cutoff_wavelength = 10000  # meters
print(f"\nApplying spectral filter with cutoff wavelength: {cutoff_wavelength/1000:.1f} km")

# Apply spectral filter using xrft
filtered_elevation_da = spectral_cutoff_filter(elevation_da, cutoff_wavelength, "x", "y")

print("Filtering completed!")

# Create 3D surface plots using xarray's native plotting
fig = plt.figure(figsize=(18, 8))

vmax = elevation_da.max().values

# Plot original bathymetry using xarray's surface plot
ax1 = fig.add_subplot(121, projection='3d')
surf1 = elevation_da.plot.surface(ax=ax1, x='x', y='y', cmap='terrain', 
                                  alpha=1, linewidth=0, antialiased=True,
                                  add_colorbar=False)
ax1.set_title('Original Bathymetry')
ax1.set_zlabel('Elevation (m)')
ax1.view_init(elev=30, azim=45)

# Plot filtered bathymetry using xarray's surface plot
ax2 = fig.add_subplot(122, projection='3d')
surf2 = filtered_elevation_da.plot.surface(ax=ax2, x='x', y='y', cmap='terrain',
                                           alpha=1, linewidth=0, antialiased=True,
                                           add_colorbar=False)
ax2.set_title(f'Filtered Bathymetry (λ > {cutoff_wavelength/1000:.1f} km)')
ax2.set_zlabel('Elevation (m)')
ax2.view_init(elev=30, azim=45)

# Set same z-limits for both plots using xarray data
z_min = min(elevation_da.min().values, filtered_elevation_da.min().values)
z_max = max(elevation_da.max().values, filtered_elevation_da.max().values)
ax1.set_zlim(z_min, z_max)
ax2.set_zlim(z_min, z_max)

# Add colorbar
plt.tight_layout()
cbar = fig.colorbar(surf1, ax=[ax1, ax2], shrink=0.6, aspect=30, pad=0.1)
cbar.set_label('Elevation (m)')

# Add overall title
fig.suptitle('3D Surface: Spectral Filtering of Seamount Bathymetry', fontsize=16, y=0.95)

plt.show()

print("\nScript completed successfully!")
