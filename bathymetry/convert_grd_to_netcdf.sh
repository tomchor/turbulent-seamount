#!/bin/bash
#
# Convert GMT GRD file to NetCDF format
#
# This script converts the GMRT GMT grid file to NetCDF format using GMT's grdconvert
#
# Usage: ./convert_grd_to_netcdf.sh
#

# Define file paths
INPUT_FILE="GMRT/GMRTv4_4_0_20250929topo.grd"
OUTPUT_FILE="GMRT/GMRTv4_4_0_20250929topo.nc"

echo "Converting $INPUT_FILE to $OUTPUT_FILE using GMT..."

# Check if GMT is available
if ! command -v gmt &> /dev/null; then
    echo "Error: GMT is not installed or not in PATH"
    echo "Please install GMT or use the Python script instead"
    exit 1
fi

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file $INPUT_FILE not found!"
    exit 1
fi

# Convert using GMT grdconvert
# =cf specifies COARDS-compliant NetCDF format
if gmt grdconvert "$INPUT_FILE" -G"$OUTPUT_FILE"=cf; then
    echo "✓ Conversion successful!"
    echo "Output file: $OUTPUT_FILE"
    echo "File size: $(du -h "$OUTPUT_FILE" | cut -f1)"
    
    # Show basic info about the converted file
    if command -v ncdump &> /dev/null; then
        echo ""
        echo "Dataset info:"
        ncdump -h "$OUTPUT_FILE" | grep -E "dimensions:|variables:"
    fi
else
    echo "❌ Conversion failed!"
    echo ""
    echo "Alternative methods:"
    echo "1. Try the Python script: python convert_grd_to_netcdf.py"
    echo "2. Use GDAL: gdal_translate -of NetCDF $INPUT_FILE $OUTPUT_FILE"
    exit 1
fi
