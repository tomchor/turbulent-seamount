import pandas as pd
import numpy as np

# Read the seamount database CSV file
# Skip the description lines and read the data
df = pd.read_csv("data/KWSMTSv01.txt", 
                 sep=r"\s+",  # whitespace separator
                 skiprows=15,  # skip the description lines
                 engine="python")

# Add column descriptions as attributes
column_descriptions = {
    'Longitude': 'Longitude (-180/+180). Center of each seamount (in degrees)',
    'Latitude': 'Latitude (-90/+90). Center of each seamount (in degrees)',
    'Azimuth': 'Estimated azimuth of the basal ellipse (in degree)',
    'Major': 'Estimated major axis of the basal ellipse of each seamount (in km)',
    'Minor': 'Estimated minor axis of the basal ellipse of each seamount (in km)',
    'Height': 'Seamount height obtained from the prediced bathymetry TOPO V12 (in m)',
    'FAA': 'Maximum amplitude of the Free-Air Gravity Anomaly (in mGal)',
    'VGG': 'Maximum amplitude of the Vertical Gravity Gradient Anomaly (in Eotvos)',
    'Depth': 'Regional depth of each seamount (in m)',
    'CrustAge': 'Age of underlying seafloor from the AGE 3.2 grid (in Myr)',
    'ID': 'ID for each seamount (plate_###)'
}

# Set column descriptions as attributes
for col in df.columns:
    if col in column_descriptions:
        setattr(df[col], 'description', column_descriptions[col])

df["width"] = np.sqrt(df.Major**2 + df.Minor**2)
df["width_meters"] = df.width * 1000
df["FWHM"] = df.width_meters / 2 # width here is the basal ellipse width, which is larger than the FWHM
df["aspect_ratio"] = df.Height / df.FWHM

df["eccentricity"] = np.sqrt(1 - (df.Minor / df.Major)**2)
df["axes_ratio"] = df.Minor / df.Major

acc = df.where(df.Latitude < -45).dropna()

# Display basic info
print("aspect ratio for all seamounts: ", df.aspect_ratio.median())
print("aspect ratio for ACC seamounts: ", acc.aspect_ratio.median())
print()
print("FWHM for all seamounts: ", df.FWHM.median())
print("FWHM for ACC seamounts: ", acc.FWHM.median())
print()
print("eccentricity for all seamounts: ", df.eccentricity.median())
print("eccentricity for ACC seamounts: ", acc.eccentricity.median())

print("axes ratio for all seamounts: ", df.axes_ratio.median())
print("axes ratio for ACC seamounts: ", acc.axes_ratio.median())