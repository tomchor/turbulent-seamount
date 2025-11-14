#!/usr/bin/env python3
"""
Convert GeoTIFF file to gridded NetCDF format with proper lat/lon coordinates

Simple script to convert GeoTIFF files to NetCDF format with proper latitude/longitude
coordinates for use with Python/xarray.

Usage:
    python convert_tiff_to_netcdf.py [input_file] [output_file]

Default: converts tiff/sars_hillshade.tiff to tiff/sars_hillshade.nc

Requirements:
    pip install rasterio xarray
"""

import os
import sys

def convert_geotiff_to_netcdf(input_file, output_file=""):
    """Convert GeoTIFF to NetCDF with proper lat/lon coordinates"""
    try:
        import rasterio
        import xarray as xr
        import numpy as np
    except ImportError as e:
        print(f"Error: Missing required package: {e}")
        print("Install with: pip install rasterio xarray")
        sys.exit(1)
    
    # Set default output file if not provided
    if not output_file:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}.nc"
    
    print(f"Converting {input_file} to {output_file}...")
    
    # Open and read the GeoTIFF
    with rasterio.open(input_file) as src:
        # Read the data (assume single band)
        data = src.read(1)
        
        # Get coordinate information
        height, width = src.shape
        transform = src.transform
        crs = src.crs
        
        # Create coordinate arrays
        x_coords = np.linspace(src.bounds.left, src.bounds.right, width)
        y_coords = np.linspace(src.bounds.top, src.bounds.bottom, height)
    
    # Create xarray Dataset
    ds = xr.Dataset({
        'data': (['y', 'x'], data)
    }, coords={
        'x': ('x', x_coords),
        'y': ('y', y_coords)
    })
    
    # If coordinates are geographic (lat/lon), rename them
    if crs and crs.is_geographic:
        ds = ds.rename({'x': 'lon', 'y': 'lat'})
        
        # Add coordinate attributes
        ds.lon.attrs = {
            'units': 'degrees_east',
            'long_name': 'longitude',
            'standard_name': 'longitude'
        }
        ds.lat.attrs = {
            'units': 'degrees_north',
            'long_name': 'latitude',
            'standard_name': 'latitude'
        }
    else:
        # Keep as x/y for projected coordinates
        ds.x.attrs = {'long_name': 'x coordinate', 'units': 'm'}
        ds.y.attrs = {'long_name': 'y coordinate', 'units': 'm'}
    
    # Add data variable attributes
    ds.data.attrs = {
        'long_name': 'GeoTIFF data',
        'source': os.path.basename(input_file)
    }
    
    # Add global attributes
    ds.attrs = {
        'title': f'Converted from {os.path.basename(input_file)}',
        'source': input_file,
        'crs': str(crs) if crs else 'unknown'
    }
    
    # Save to NetCDF
    ds.to_netcdf(output_file)
    print(f"✓ Successfully created {output_file}")
    
    # Show basic info
    print(f"Dimensions: {dict(ds.dims)}")
    print(f"Coordinates: {list(ds.coords)}")
    
    return output_file

def main():
    """Main function - handles command line arguments or uses defaults"""
    if len(sys.argv) == 1:
        # No arguments - use default files
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_file = os.path.join(script_dir, "tiff", "sars_hillshade.tiff")
        output_file = ""
    elif len(sys.argv) == 2:
        # One argument - input file provided
        input_file = sys.argv[1]
        output_file = ""
    elif len(sys.argv) == 3:
        # Two arguments - both files provided
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        print("Usage: python convert_tiff_to_netcdf.py [input_file] [output_file]")
        sys.exit(1)
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found!")
        sys.exit(1)
    
    # Convert the file
    try:
        output_path = convert_geotiff_to_netcdf(input_file, output_file)
        print(f"Conversion complete!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
