#!/usr/bin/env python3
"""
Convert GMT GRD file to NetCDF format with proper lat/lon gridding

This script converts the GMRT GMT grid file to NetCDF format that can be used
with xarray and other Python tools, ensuring the output has proper gridded
latitude and longitude coordinates.

Usage:
    python convert_grd_to_netcdf.py

Input:  GMRT/GMRTv4_4_0_20250929topo.grd
Output: GMRT/GMRTv4_4_0_20250929topo.nc
"""

import os
import subprocess
import sys
import tempfile

def check_gmt_available():
    """Check if GMT is available on the system"""
    try:
        result = subprocess.run(['gmt', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"GMT version found: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("GMT not found on system. Please install GMT or use alternative method.")
        return False

def get_grid_info(input_file):
    """Get information about the grid using GMT grdinfo"""
    try:
        cmd = ['gmt', 'grdinfo', str(input_file)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Could not get grid info: {e}")
        return None

def convert_with_gmt(input_file, output_file):
    """Convert GRD to NetCDF using GMT's grdconvert with proper lat/lon gridding"""
    try:
        # First, get information about the input grid
        print("Getting grid information...")
        grid_info = get_grid_info(input_file)
        if grid_info:
            print("Input grid info:")
            print(grid_info)
        
        # Convert to NetCDF format with COARDS compliance for proper lat/lon structure
        print("Converting to NetCDF format...")
        cmd = ['gmt', 'grdconvert', str(input_file), f'-G{output_file}=nf']
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Successfully converted {input_file} to {output_file} using GMT")
        
        # Post-process to ensure proper coordinate naming and structure
        return post_process_netcdf(output_file)
        
    except subprocess.CalledProcessError as e:
        print(f"GMT conversion failed: {e}")
        print(f"GMT stderr: {e.stderr}")
        return False

def post_process_netcdf(output_file):
    """Post-process NetCDF to ensure proper lat/lon coordinate structure"""
    try:
        # Try to load and verify the NetCDF structure
        import xarray as xr
        print("Post-processing NetCDF file to ensure proper lat/lon gridding...")
        
        # Open the file and check structure
        ds = xr.open_dataset(output_file)
        print(f"Original dimensions: {list(ds.dims)}")
        print(f"Original coordinates: {list(ds.coords)}")
        print(f"Original variables: {list(ds.data_vars)}")
        
        # Check if we need to rename coordinates to standard names
        needs_update = False
        new_ds = ds.copy()
        
        # Common coordinate name mappings
        coord_mappings = {
            'x': 'lon',
            'longitude': 'lon', 
            'y': 'lat',
            'latitude': 'lat'
        }
        
        # Rename coordinates if needed
        for old_name, new_name in coord_mappings.items():
            if old_name in ds.dims and new_name not in ds.dims:
                new_ds = new_ds.rename({old_name: new_name})
                needs_update = True
                print(f"Renamed dimension '{old_name}' to '{new_name}'")
        
        # Ensure proper coordinate attributes
        if 'lon' in new_ds.coords:
            new_ds.lon.attrs.update({
                'units': 'degrees_east',
                'long_name': 'longitude',
                'standard_name': 'longitude'
            })
            needs_update = True
            
        if 'lat' in new_ds.coords:
            new_ds.lat.attrs.update({
                'units': 'degrees_north',
                'long_name': 'latitude', 
                'standard_name': 'latitude'
            })
            needs_update = True
        
        # Find the main data variable and ensure proper attributes
        data_vars = [var for var in new_ds.data_vars if var not in ['lat', 'lon']]
        if data_vars:
            main_var = data_vars[0]  # Assume first data variable is elevation
            if main_var in new_ds:
                new_ds[main_var].attrs.update({
                    'units': 'm',
                    'long_name': 'elevation',
                    'standard_name': 'height_above_reference_ellipsoid'
                })
                needs_update = True
        
        # Add global attributes
        new_ds.attrs.update({
            'title': 'GMRT Bathymetry Data',
            'source': 'Global Multi-Resolution Topography (GMRT)',
            'Conventions': 'CF-1.6'
        })
        needs_update = True
        
        if needs_update:
            print("Saving updated NetCDF file...")
            # Close original dataset
            ds.close()
            
            # Save the updated version
            new_ds.to_netcdf(output_file)
            new_ds.close()
            print("✓ Post-processing completed successfully")
        else:
            ds.close()
            new_ds.close()
            print("✓ File already has proper structure")
            
        return True
        
    except ImportError:
        print("Warning: xarray not available for post-processing")
        print("File converted but may need manual coordinate verification")
        return True
    except Exception as e:
        print(f"Warning: Post-processing failed: {e}")
        print("File converted but structure may need manual verification")
        return True

def try_xarray_conversion(input_file, output_file):
    """Try to convert using xarray with proper lat/lon gridding"""
    try:
        import xarray as xr
        print(f"Attempting to open {input_file} with xarray...")
        
        # Try to open with xarray
        ds = xr.open_dataset(input_file)
        print("Successfully opened with xarray!")
        
        # Post-process to ensure proper gridding before saving
        ds_processed = ensure_proper_gridding(ds)
        
        # Save as NetCDF
        ds_processed.to_netcdf(output_file)
        print(f"Successfully converted {input_file} to {output_file} using xarray")
        ds.close()
        ds_processed.close()
        return True
        
    except ImportError:
        print("xarray not available, skipping xarray method")
        return False
    except Exception as e:
        print(f"Xarray conversion failed: {e}")
        return False

def ensure_proper_gridding(ds):
    """Ensure dataset has proper lat/lon gridding structure"""
    import xarray as xr
    
    # Create a copy to modify
    new_ds = ds.copy()
    
    # Common coordinate name mappings
    coord_mappings = {
        'x': 'lon',
        'longitude': 'lon', 
        'y': 'lat',
        'latitude': 'lat'
    }
    
    # Rename coordinates if needed
    for old_name, new_name in coord_mappings.items():
        if old_name in ds.dims and new_name not in ds.dims:
            new_ds = new_ds.rename({old_name: new_name})
            print(f"Renamed dimension '{old_name}' to '{new_name}'")
    
    # Ensure coordinates are properly ordered (lat decreasing, lon increasing typically)
    if 'lat' in new_ds.dims:
        # Sort latitude in descending order if not already
        if len(new_ds.lat) > 1 and new_ds.lat[0] < new_ds.lat[-1]:
            new_ds = new_ds.isel(lat=slice(None, None, -1))
            print("Reordered latitude coordinates (descending)")
    
    if 'lon' in new_ds.dims:
        # Sort longitude in ascending order if not already
        if len(new_ds.lon) > 1 and new_ds.lon[0] > new_ds.lon[-1]:
            new_ds = new_ds.isel(lon=slice(None, None, -1))
            print("Reordered longitude coordinates (ascending)")
    
    # Ensure proper coordinate attributes
    if 'lon' in new_ds.coords:
        new_ds.lon.attrs.update({
            'units': 'degrees_east',
            'long_name': 'longitude',
            'standard_name': 'longitude'
        })
        
    if 'lat' in new_ds.coords:
        new_ds.lat.attrs.update({
            'units': 'degrees_north',
            'long_name': 'latitude', 
            'standard_name': 'latitude'
        })
    
    # Find and set attributes for data variables
    data_vars = [var for var in new_ds.data_vars if var not in ['lat', 'lon']]
    for var_name in data_vars:
        if 'z' in var_name.lower() or 'elev' in var_name.lower() or 'topo' in var_name.lower():
            new_ds[var_name].attrs.update({
                'units': 'm',
                'long_name': 'elevation',
                'standard_name': 'height_above_reference_ellipsoid'
            })
    
    # Add global attributes
    new_ds.attrs.update({
        'title': 'GMRT Bathymetry Data',
        'source': 'Global Multi-Resolution Topography (GMRT)',
        'Conventions': 'CF-1.6'
    })
    
    return new_ds

def main():
    """Main conversion function"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define file paths
    input_file = os.path.join(script_dir, "GMRT", "GMRTv4_4_0_20250929topo.grd")
    output_file = os.path.join(script_dir, "GMRT", "GMRTv4_4_0_20250929topo.nc")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found!")
        sys.exit(1)
    
    print(f"Converting {input_file} to {output_file}")
    
    # Try GMT first (preferred method for GRD files)
    if check_gmt_available():
        if convert_with_gmt(input_file, output_file):
            # Verify the output file
            if os.path.exists(output_file):
                print(f"✓ Conversion successful!")
                print(f"Output file: {output_file}")
                file_size_mb = os.path.getsize(output_file) / (1024*1024)
                print(f"File size: {file_size_mb:.1f} MB")
                
                # Try to inspect the converted file
                try:
                    import xarray as xr
                    ds = xr.open_dataset(output_file)
                    print("\n" + "="*50)
                    print("FINAL NETCDF STRUCTURE:")
                    print("="*50)
                    print(f"Dimensions: {dict(ds.dims)}")
                    print(f"Coordinates: {list(ds.coords)}")
                    print(f"Data Variables: {list(ds.data_vars)}")
                    
                    # Show coordinate ranges
                    if 'lat' in ds.coords:
                        lat_range = (ds.lat.min().item(), ds.lat.max().item())
                        print(f"Latitude range: {lat_range[0]:.4f}° to {lat_range[1]:.4f}°")
                    if 'lon' in ds.coords:
                        lon_range = (ds.lon.min().item(), ds.lon.max().item())
                        print(f"Longitude range: {lon_range[0]:.4f}° to {lon_range[1]:.4f}°")
                    
                    # Show elevation/bathymetry info
                    data_vars = [var for var in ds.data_vars if var not in ['lat', 'lon']]
                    if data_vars:
                        main_var = data_vars[0]
                        elev_range = (ds[main_var].min().item(), ds[main_var].max().item())
                        print(f"{main_var} range: {elev_range[0]:.1f}m to {elev_range[1]:.1f}m")
                    
                    # Verify gridding structure
                    if 'lat' in ds.dims and 'lon' in ds.dims:
                        grid_shape = (ds.dims['lat'], ds.dims['lon'])
                        print(f"Grid shape (lat, lon): {grid_shape}")
                        print("✓ File has proper lat/lon gridded structure")
                    else:
                        print("⚠ Warning: File may not have standard lat/lon grid structure")
                    
                    print("="*50)
                    ds.close()
                except ImportError:
                    print("(xarray not available for file inspection)")
                except Exception as e:
                    print(f"Warning: Could not inspect converted file: {e}")
                
                return
    
    # Fallback to xarray method
    print("Trying xarray method as fallback...")
    if try_xarray_conversion(input_file, output_file):
        print(f"✓ Conversion successful with xarray!")
        return
    
    # If both methods fail, provide manual instructions
    print("\n❌ Automatic conversion failed.")
    print("\nManual conversion options:")
    print("1. Install GMT and run:")
    print(f"   gmt grdconvert {input_file} -G{output_file}=cf")
    print("\n2. Use GDAL tools:")
    print(f"   gdal_translate -of NetCDF {input_file} {output_file}")
    print("\n3. Or check if the file is already in a compatible format")
    
    sys.exit(1)

if __name__ == "__main__":
    main()
