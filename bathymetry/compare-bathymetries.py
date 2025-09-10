import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

def load_preprocessed_bathymetry():
    """Load the preprocessed bathymetry data"""
    ds_prep = xr.open_dataset("balanus-bathymetry-preprocessed.nc")
    return ds_prep

def load_gebco_bathymetry():
    """Load the GEBCO bathymetry data"""
    ds_gebco = xr.open_dataset("GEBCO/balanus-gebco_2024_n39.8_s39.0_w-65.8_e-65.0.nc")
    return ds_gebco

def convert_gebco_to_meters(ds_gebco, reference_lat=39.4):
    """
    Convert GEBCO coordinates from degrees to meters, centered on the seamount
    
    Parameters:
    -----------
    ds_gebco : xarray.Dataset
        GEBCO dataset with lat/lon coordinates
    reference_lat : float
        Reference latitude for longitude-to-meter conversion
    """
    # Find the peak (maximum elevation) in GEBCO data
    peak_idx = ds_gebco.elevation.argmax()
    peak_coords = np.unravel_index(peak_idx.item(), ds_gebco.elevation.shape)
    peak_lat = ds_gebco.lat.values[peak_coords[0]]
    peak_lon = ds_gebco.lon.values[peak_coords[1]]
    
    # Conversion factors
    lat2meter = 111.3e3  # meters per degree latitude
    lon2meter = lat2meter * np.cos(np.deg2rad(reference_lat))  # meters per degree longitude
    
    # Convert to meters and center on peak
    x_meters = (ds_gebco.lon.values - peak_lon) * lon2meter
    y_meters = (ds_gebco.lat.values - peak_lat) * lat2meter
    
    # Create new dataset with converted coordinates
    ds_converted = ds_gebco.copy()
    ds_converted = ds_converted.assign_coords(
        lon=("lon", x_meters),
        lat=("lat", y_meters)
    )
    ds_converted = ds_converted.rename({"lon": "x", "lat": "y"})
    
    return ds_converted

def plot_bathymetry_comparison():
    """Create side-by-side comparison plots of both bathymetries"""
    
    # Load datasets
    ds_prep = load_preprocessed_bathymetry()
    ds_gebco = load_gebco_bathymetry()
    ds_gebco_meters = convert_gebco_to_meters(ds_gebco)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Bathymetry Comparison: Preprocessed vs GEBCO', fontsize=16, fontweight='bold')
    
    # Define common colormap and limits for elevation
    vmin = min(ds_prep.z.min().item(), ds_gebco_meters.elevation.min().item())
    vmax = max(ds_prep.z.max().item(), ds_gebco_meters.elevation.max().item())
    
    # Plot 1: Preprocessed bathymetry (z variable)
    im1 = axes[0,0].contourf(ds_prep.x/1000, ds_prep.y/1000, ds_prep.z, 
                            levels=50, cmap='terrain', vmin=vmin, vmax=vmax)
    axes[0,0].set_title('Preprocessed Bathymetry\n(z elevation)', fontweight='bold')
    axes[0,0].set_xlabel('X Distance (km)')
    axes[0,0].set_ylabel('Y Distance (km)')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_aspect('equal')
    
    # Plot 2: GEBCO bathymetry
    im2 = axes[0,1].contourf(ds_gebco_meters.x/1000, ds_gebco_meters.y/1000, ds_gebco_meters.elevation,
                            levels=50, cmap='terrain', vmin=vmin, vmax=vmax)
    axes[0,1].set_title('GEBCO Bathymetry\n(elevation)', fontweight='bold')
    axes[0,1].set_xlabel('X Distance (km)')
    axes[0,1].set_ylabel('Y Distance (km)')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_aspect('equal')
    
    # Plot 3: Preprocessed periodic elevation
    im3 = axes[0,2].contourf(ds_prep.x/1000, ds_prep.y/1000, ds_prep.periodic_elevation,
                            levels=50, cmap='terrain')
    axes[0,2].set_title('Preprocessed Bathymetry\n(periodic elevation)', fontweight='bold')
    axes[0,2].set_xlabel('X Distance (km)')
    axes[0,2].set_ylabel('Y Distance (km)')
    axes[0,2].grid(True, alpha=0.3)
    axes[0,2].set_aspect('equal')
    
    # Add colorbars
    cbar1 = plt.colorbar(im1, ax=axes[0,0], shrink=0.8)
    cbar1.set_label('Elevation (m)', rotation=270, labelpad=20)
    
    cbar2 = plt.colorbar(im2, ax=axes[0,1], shrink=0.8)
    cbar2.set_label('Elevation (m)', rotation=270, labelpad=20)
    
    cbar3 = plt.colorbar(im3, ax=axes[0,2], shrink=0.8)
    cbar3.set_label('Elevation (m)', rotation=270, labelpad=20)
    
    # Plot 4: Cross-section comparison
    # Take a cross-section through the center (y=0)
    y_center_prep = ds_prep.y.values[np.argmin(np.abs(ds_prep.y.values))]
    y_center_gebco = ds_gebco_meters.y.values[np.argmin(np.abs(ds_gebco_meters.y.values))]
    
    cross_section_prep = ds_prep.z.sel(y=y_center_prep, method='nearest')
    cross_section_gebco = ds_gebco_meters.elevation.sel(y=y_center_gebco, method='nearest')
    
    axes[1,0].plot(cross_section_prep.x/1000, cross_section_prep, 'b-', linewidth=2, label='Preprocessed')
    axes[1,0].plot(cross_section_gebco.x/1000, cross_section_gebco, 'r-', linewidth=2, label='GEBCO')
    axes[1,0].set_title('Cross-section Comparison (y≈0)', fontweight='bold')
    axes[1,0].set_xlabel('X Distance (km)')
    axes[1,0].set_ylabel('Elevation (m)')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 5: Statistics comparison
    axes[1,1].axis('off')
    
    # Calculate statistics
    prep_stats = {
        'Min': ds_prep.z.min().item(),
        'Max': ds_prep.z.max().item(),
        'Mean': ds_prep.z.mean().item(),
        'Std': ds_prep.z.std().item(),
        'Height': ds_prep.attrs.get('H', 'N/A'),
        'FWHM': ds_prep.attrs.get('FWHM', 'N/A')
    }
    
    gebco_stats = {
        'Min': ds_gebco_meters.elevation.min().item(),
        'Max': ds_gebco_meters.elevation.max().item(),
        'Mean': ds_gebco_meters.elevation.mean().item(),
        'Std': ds_gebco_meters.elevation.std().item(),
        'Height': ds_gebco_meters.elevation.max().item() - ds_gebco_meters.elevation.min().item(),
        'FWHM': 'N/A'
    }
    
    # Create statistics table
    stats_text = "DATASET STATISTICS\n" + "="*30 + "\n\n"
    stats_text += f"{'Metric':<15} {'Preprocessed':<15} {'GEBCO':<15}\n"
    stats_text += "-" * 45 + "\n"
    
    for key in ['Min', 'Max', 'Mean', 'Std']:
        if isinstance(prep_stats[key], (int, float)) and isinstance(gebco_stats[key], (int, float)):
            stats_text += f"{key:<15} {prep_stats[key]:<15.1f} {gebco_stats[key]:<15.1f}\n"
        else:
            stats_text += f"{key:<15} {str(prep_stats[key]):<15} {str(gebco_stats[key]):<15}\n"

    stats_text += f"Height      {prep_stats['Height']:<15.1f} {gebco_stats['Height']:<15.1f}\n"
    stats_text += f"FWHM           {prep_stats['FWHM']:<15} {gebco_stats['FWHM']:<15}\n"
    
    axes[1,1].text(0.05, 0.95, stats_text, transform=axes[1,1].transAxes, 
                   fontfamily='monospace', fontsize=10, verticalalignment='top')
    
    # Plot 6: Original GEBCO with geographic coordinates
    axes[1,2].contourf(ds_gebco.lon, ds_gebco.lat, ds_gebco.elevation,
                      levels=50, cmap='terrain')
    axes[1,2].set_title('GEBCO Bathymetry\n(Geographic Coordinates)', fontweight='bold')
    axes[1,2].set_xlabel('Longitude (°)')
    axes[1,2].set_ylabel('Latitude (°)')
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return ds_prep, ds_gebco, ds_gebco_meters


ds_prep, ds_gebco, ds_gebco_meters = plot_bathymetry_comparison()

print("Bathymetry comparison complete!")
print(f"Preprocessed dataset shape: {ds_prep.z.shape}")
print(f"GEBCO dataset shape: {ds_gebco.elevation.shape}")
print(f"Preprocessed seamount height: {ds_prep.attrs.get('H', 'N/A')} m")
print(f"GEBCO seamount height: {ds_gebco_meters.elevation.max().item() - ds_gebco_meters.elevation.min().item():.1f} m")