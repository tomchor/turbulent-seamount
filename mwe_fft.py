import numpy as np
import xarray as xr
import xrft
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Set up the figure for better display
plt.rcParams["figure.constrained_layout.use"] = True

print("=== 3D Field with 2D FFT Analysis using xrft ===")
print("Creating 3D field (x, y, z) where only x and y are FFT transformed...")

# Define grid parameters
Lx, Ly, Lz = 100.0, 100.0, 50.0  # Domain size
Nx, Ny, Nz = 128, 128, 10         # Grid points (smaller for 3D)

# Create coordinate arrays
x = np.linspace(-Lx/2, Lx/2, Nx)
y = np.linspace(-Ly/2, Ly/2, Ny)
z = np.linspace(0, Lz, Nz)  # z from 0 to Lz

# Create xarray dataset with coordinates
coords = {'x': x, 'y': y, 'z': z}

# Create 3D meshgrid
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Define various wave components that vary with z
k1, k2 = 2*np.pi/20, 2*np.pi/15  # wavenumbers in x and y
k3, k4 = 2*np.pi/8, 2*np.pi/12   # additional wavenumbers

# Create a field with multiple sine and cosine components that vary with z
# Each z level has slightly different characteristics
field = np.zeros((Nx, Ny, Nz))
for i, z_val in enumerate(z):
    # Amplitude varies with z (like a vertical mode)
    amp_factor = 1 + 0.5 * np.sin(2*np.pi*z_val/Lz)
    
    # Create 2D pattern for this z level
    field[:, :, i] = amp_factor * (
        np.sin(k1 * X[:, :, i]) * np.cos(k2 * Y[:, :, i]) +          # Standing wave pattern
        0.5 * np.cos(k3 * X[:, :, i] + k4 * Y[:, :, i]) +            # Diagonal wave
        0.3 * np.sin(2*k1 * X[:, :, i]) * np.sin(2*k2 * Y[:, :, i]) + # Higher harmonics
        0.2 * np.cos(k1 * X[:, :, i] - k2 * Y[:, :, i]) +            # Another diagonal component
        0.1 * np.random.randn(Nx, Ny)                                # Small amount of noise
    )

# Create xarray DataArray with 3D structure
da = xr.DataArray(
    field, 
    coords=coords, 
    dims=['x', 'y', 'z'],
    name='field_3d',
    attrs={'long_name': '3D field with sines and cosines', 'units': 'arbitrary'}
)

print(f"Created 3D field with shape: {da.shape}")
print(f"Field coordinates: {list(da.coords.keys())}")
print(f"Field dimensions: {da.dims}")

print("\nPerforming 2D FFT on x and y dimensions only...")
# Perform 2D FFT with proper phase and amplitude scaling
# Note: only transforming x and y dimensions, z is preserved
da_fft = xrft.fft(
    da, 
    dim=['x', 'y'],  # Only transform x and y, NOT z
    true_phase=True, 
    true_amplitude=True
)

# Calculate power spectrum (magnitude squared)
power_spectrum = (da_fft * np.conjugate(da_fft)).real
power_spectrum.name = 'power_spectrum_3d'

print(f"FFT result shape: {da_fft.shape}")
print(f"Frequency coordinates: {list(da_fft.coords.keys())}")
print(f"FFT dimensions: {da_fft.dims}")

print("\nComputing isotropic spectrum for each z level...")
# Use xrft's isotropic power spectrum function which handles FFT internally
# This will create an isotropic spectrum for each z level
iso_spectrum = xrft.isotropic_power_spectrum(
    da, 
    dim=['x', 'y'],        # Dimensions to transform (only x and y)
    scaling='density',     # Power spectral density
    nfactor=4,            # Number of radial bins relative to data size
    truncate=True,        # Truncate at Nyquist frequency
    true_phase=True,      # Preserve phase information
    true_amplitude=True   # Proper amplitude scaling
)

print(f"Isotropic spectrum shape: {iso_spectrum.shape}")
print(f"Isotropic spectrum dimensions: {iso_spectrum.dims}")

print("\nPlotting results...")

# Create plots showing different z levels
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

# Select a few z levels for visualization
z_levels = [0, Nz//2, Nz-1]  # First, middle, and last z levels

for i, z_idx in enumerate(z_levels):
    z_val = da.z.values[z_idx]
    
    # Plot original field at this z level
    da.isel(z=z_idx).plot(ax=axes[0, i], cmap='RdBu_r')
    axes[0, i].set_title(f'Original Field at z={z_val:.1f}')
    axes[0, i].set_aspect('equal')
    
    # Plot 2D power spectrum at this z level (log scale)
    power_spec_z = power_spectrum.isel(z=z_idx)
    # Shift frequencies to center zero frequency and sort coordinates
    power_spectrum_shifted = power_spec_z.roll(
        shifts={dim: power_spec_z.sizes[dim]//2 for dim in ['freq_x', 'freq_y']}, 
        roll_coords=True
    )
    # Sort by coordinates to ensure monotonic ordering for plotting
    for dim in ['freq_x', 'freq_y']:
        power_spectrum_shifted = power_spectrum_shifted.sortby(dim)

    im = power_spectrum_shifted.plot(
        ax=axes[1, i], 
        norm=mcolors.LogNorm(vmin=1e-6, vmax=power_spectrum.max().values), 
        cmap='viridis'
    )
    axes[1, i].set_title(f'2D Power Spectrum at z={z_val:.1f}')
    axes[1, i].set_aspect('equal')
    
    # Plot isotropic spectrum at this z level
    iso_spectrum.isel(z=z_idx).plot(
        ax=axes[2, i], 
        yscale='log', 
        marker='o', 
        linestyle='-', 
        linewidth=2,
        label=f'z={z_val:.1f}'
    )
    axes[2, i].set_title(f'Isotropic Spectrum at z={z_val:.1f}')
    axes[2, i].set_xlabel('Wavenumber magnitude')
    axes[2, i].set_ylabel('Power Spectral Density')
    axes[2, i].grid(True, alpha=0.3)

# Note: constrained_layout is already enabled globally
plt.show()

# Additional plot: Compare isotropic spectra across z levels
plt.figure(figsize=(10, 6))
for i, z_idx in enumerate(z_levels):
    z_val = da.z.values[z_idx]
    iso_spectrum.isel(z=z_idx).plot(
        yscale='log', 
        marker='o', 
        linestyle='-', 
        linewidth=2,
        label=f'z={z_val:.1f}'
    )

plt.title('Isotropic Spectra Comparison Across Z Levels')
plt.xlabel('Wavenumber magnitude')
plt.ylabel('Power Spectral Density')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Print some statistics
print("\n=== Summary Statistics ===")
print(f"Original 3D field - min: {da.min().values:.3f}, max: {da.max().values:.3f}")
print(f"Power spectrum - min: {power_spectrum.min().values:.2e}, max: {power_spectrum.max().values:.2e}")
print(f"Isotropic spectrum - min: {iso_spectrum.min().values:.2e}, max: {iso_spectrum.max().values:.2e}")
print(f"Number of z levels: {len(da.z)}")

print("\nExample completed successfully!")
print("The 3D field was created with varying patterns at different z levels.")
print("2D FFT was performed only on x and y dimensions, preserving the z structure.")
print("Isotropic spectra show how the power distribution varies with depth/height.")