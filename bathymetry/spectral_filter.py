import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
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
    # Perform 2D FFT using xrft
    data_fft = xrft.fft(data, dim=[y_coord, x_coord], real_dim=x_coord)

    # Get frequency coordinates
    freq_x = data_fft.coords[f'freq_{x_coord}']
    freq_y = data_fft.coords[f'freq_{y_coord}']

    # Create 2D frequency grids
    FREQ_X, FREQ_Y = xr.broadcast(freq_x, freq_y)

    # Calculate wavenumber magnitude (frequency = wavenumber / 2π)
    K = 2 * np.pi * np.sqrt(FREQ_X**2 + FREQ_Y**2)

    # Calculate cutoff wavenumber (2π/wavelength)
    k_cutoff = 2 * np.pi / cutoff_wavelength

    # Create sharp filter (1 for wavelengths > cutoff, 0 for wavelengths < cutoff)
    filter_mask = K <= k_cutoff

    # Apply filter in frequency domain
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
x = ds["x"].values
y = ds["y"].values

# Get grid spacing for information
dx = np.diff(x).mean()
dy = np.diff(y).mean()

print(f"Grid dimensions: {elevation_da.shape}")
print(f"Grid spacing: dx = {dx:.1f} m, dy = {dy:.1f} m")
print(f"Domain size: {(x.max() - x.min())/1000:.1f} km × {(y.max() - y.min())/1000:.1f} km")
print(f"Elevation range: {elevation_da.min().values:.1f} to {elevation_da.max().values:.1f} m")

# Define cutoff wavelength (you can adjust this)
# For example, filter out wavelengths shorter than 10 km
cutoff_wavelength = 10000  # meters

print(f"\nApplying spectral filter with cutoff wavelength: {cutoff_wavelength/1000:.1f} km")

# Apply spectral filter using xrft
filtered_elevation_da = spectral_cutoff_filter(elevation_da, cutoff_wavelength, "x", "y")

# Convert back to numpy arrays for plotting
elevation = elevation_da.values
filtered_elevation = filtered_elevation_da.values

print("Filtering completed!")

# Create side-by-side plot
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Common colormap settings
vmin = 0
vmax = max(elevation.max(), filtered_elevation.max())

# Plot original bathymetry
im1 = axes[0].contourf(x/1000, y/1000, elevation, levels=50, cmap='terrain',
                        vmin=vmin, vmax=vmax, extend='both')
axes[0].set_title('Original Bathymetry')
axes[0].set_xlabel('X (km)')
axes[0].set_ylabel('Y (km)')
axes[0].set_aspect('equal')

# Plot filtered bathymetry
im2 = axes[1].contourf(x/1000, y/1000, filtered_elevation, levels=50, cmap='terrain',
                        vmin=vmin, vmax=vmax, extend='both')
axes[1].set_title(f'Filtered Bathymetry (λ > {cutoff_wavelength/1000:.1f} km)')
axes[1].set_xlabel('X (km)')
axes[1].set_ylabel('Y (km)')
axes[1].set_aspect('equal')

# Add colorbar
plt.tight_layout()
cbar = fig.colorbar(im1, ax=axes, shrink=0.8, aspect=30)
cbar.set_label('Elevation (m)')

# Add overall title
fig.suptitle('Spectral Filtering of Seamount Bathymetry', fontsize=16, y=0.95)

plt.show()

# Calculate and display some statistics
print(f"\nStatistics:")
print(f"Original elevation - min: {elevation.min():.1f} m, max: {elevation.max():.1f} m, std: {elevation.std():.1f} m")
print(f"Filtered elevation - min: {filtered_elevation.min():.1f} m, max: {filtered_elevation.max():.1f} m, std: {filtered_elevation.std():.1f} m")

# Calculate difference
difference = elevation - filtered_elevation
print(f"Difference (original - filtered) - min: {difference.min():.1f} m, max: {difference.max():.1f} m, std: {difference.std():.1f} m")

# Optional: Create a third plot showing the difference
create_difference_plot = True
if create_difference_plot:
    fig2, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.contourf(x/1000, y/1000, difference, levels=50, cmap='RdBu_r', extend='both')
    ax.set_title('Difference: Original - Filtered Bathymetry')
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_aspect('equal')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Elevation Difference (m)')
    plt.tight_layout()
    plt.show()

# Save filtered data (optional)
save_filtered_data = False
if save_filtered_data:
    # Create new dataset with filtered data
    ds_filtered = ds.copy()
    ds_filtered["filtered_elevation"] = (["y", "x"], filtered_elevation)
    ds_filtered.attrs["filter_cutoff_wavelength_m"] = cutoff_wavelength
    ds_filtered.attrs["filter_type"] = "sharp_spectral_cutoff"

    output_file = f"balanus-bathymetry-filtered_{cutoff_wavelength/1000:.1f}km.nc"
    ds_filtered.to_netcdf(output_file)
    print(f"\nFiltered data saved to: {output_file}")

# Close the dataset
ds.close()

print("\nScript completed successfully!")
