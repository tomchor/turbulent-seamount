#!/bin/bash
#
# Convert TIFF file to NetCDF format using GDAL
#
# This script converts geospatial TIFF files to NetCDF format using GDAL tools
#
# Usage: ./convert_tiff_to_netcdf.sh [input_file] [output_file]
# Default: converts tiff/sars_hillshade.tiff to tiff/sars_hillshade.nc
#

# Set default file paths
INPUT_FILE="${1:-tiff/sars_hillshade.tiff}"
OUTPUT_FILE="${2:-tiff/sars_hillshade.nc}"

echo "Converting TIFF to NetCDF using GDAL..."
echo "Input:  $INPUT_FILE"
echo "Output: $OUTPUT_FILE"

# Check if GDAL is available
if ! command -v gdal_translate &> /dev/null; then
    echo "Error: GDAL is not installed or not in PATH"
    echo "Please install GDAL or use the Python script instead"
    echo ""
    echo "On Ubuntu/Debian: sudo apt-get install gdal-bin"
    echo "On macOS: brew install gdal"
    echo "Or use conda: conda install gdal"
    exit 1
fi

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file $INPUT_FILE not found!"
    exit 1
fi

# Show input file info
echo ""
echo "Input file information:"
if command -v gdalinfo &> /dev/null; then
    gdalinfo "$INPUT_FILE" | head -20
else
    echo "gdalinfo not available for file inspection"
fi

echo ""
echo "Converting..."

# Convert TIFF to NetCDF using gdal_translate
if gdal_translate -of NetCDF "$INPUT_FILE" "$OUTPUT_FILE"; then
    echo "✓ Conversion successful!"
    echo "Output file: $OUTPUT_FILE"
    echo "File size: $(du -h "$OUTPUT_FILE" | cut -f1)"
    
    # Show basic info about the converted file
    echo ""
    echo "Output file information:"
    if command -v ncdump &> /dev/null; then
        echo "Dimensions and variables:"
        ncdump -h "$OUTPUT_FILE" | grep -E "dimensions:|variables:" -A 5
        echo ""
        echo "Coordinate ranges:"
        ncdump -v x,y "$OUTPUT_FILE" 2>/dev/null | grep -E "x = |y = " | head -2
    elif command -v gdalinfo &> /dev/null; then
        gdalinfo "$OUTPUT_FILE" | head -15
    fi
else
    echo "❌ Conversion failed!"
    echo ""
    echo "Alternative method:"
    echo "Try the Python script: python convert_tiff_to_netcdf.py"
    exit 1
fi

echo ""
echo "Note: The converted NetCDF may need coordinate renaming for lat/lon."
echo "Use the Python script for more advanced coordinate handling."
