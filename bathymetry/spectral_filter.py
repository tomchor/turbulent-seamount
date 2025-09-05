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

# Plot with km coordinates
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

vmax = elevation_da.max().values

# Plot original bathymetry
im1 = elevation_da.plot.contourf(ax=axes[0], x='x', y='y', levels=50, cmap='terrain',
                                   vmin=0, vmax=vmax, extend='both', add_colorbar=False)
axes[0].set_title('Original Bathymetry')
axes[0].set_aspect('equal')

# Plot filtered bathymetry
im2 = filtered_elevation_da.plot.contourf(ax=axes[1], x='x', y='y', levels=50, cmap='terrain',
                                            vmin=0, vmax=vmax, extend='both', add_colorbar=False)
axes[1].set_title(f'Filtered Bathymetry (λ > {cutoff_wavelength/1000:.1f} km)')
axes[1].set_aspect('equal')

# Add colorbar
plt.tight_layout()
cbar = fig.colorbar(im1, ax=axes, shrink=0.8, aspect=30)
cbar.set_label('Elevation (m)')

# Add overall title
fig.suptitle('Spectral Filtering of Seamount Bathymetry', fontsize=16, y=0.95)

plt.show()

# Calculate and display some statistics using xarray
print(f"\nStatistics:")
print(f"Original elevation - min: {elevation_da.min().values:.1f} m, max: {elevation_da.max().values:.1f} m, std: {elevation_da.std().values:.1f} m")
print(f"Filtered elevation - min: {filtered_elevation_da.min().values:.1f} m, max: {filtered_elevation_da.max().values:.1f} m, std: {filtered_elevation_da.std().values:.1f} m")

# Calculate difference using xarray
difference_da = elevation_da - filtered_elevation_da
print(f"Difference (original - filtered) - min: {difference_da.min().values:.1f} m, max: {difference_da.max().values:.1f} m, std: {difference_da.std().values:.1f} m")

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
