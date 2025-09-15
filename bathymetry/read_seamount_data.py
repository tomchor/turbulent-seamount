# This script reads the seamount_data.mat file, which can be downloaded from
# https://zenodo.org/records/11145852/files/seamount_data.mat?download=1
# and which is part of the data made available by Mashayek et al. (2024):
# https://doi.org/10.1073/pnas.2322163121

# Data contains fields of length 23353, corresponding to seamounts in Kim & Wessel 2011 (KW2011)
# seamount census. Seamounts have been removed if the half-height depth is above 1000m.

import numpy as np
from scipy.io import loadmat
import pandas as pd

# Read the MATLAB file
print("Reading seamount_data.mat file...")
seamount_data = loadmat("../bathymetry/seamount_data.mat")

# Display available fields (excluding MATLAB metadata)
print("Available fields in seamount_data.mat:")
for key in sorted(seamount_data.keys()):
    if not key.startswith("__"):  # Skip MATLAB metadata
        data = seamount_data[key]
        if isinstance(data, np.ndarray):
            print(f"  {key}: {type(data)} of shape {data.shape}")
        else:
            print(f"  {key}: {type(data)}")

# Extract key variables (flatten 2D arrays if needed)
def extract_var(name):
    var = seamount_data[name]
    return var.flatten() if var.ndim > 1 else var

# Variable descriptions from seamount_data_info.txt:
sm_id = seamount_data["sm_id"]  # seamount identifier (KW2011)
sm_lon = extract_var("sm_lon")  # seamount longitude (KW2011)
sm_lat = extract_var("sm_lat")  # seamount latitude (KW2011)
sm_depth = extract_var("sm_depth")  # depth of ocean surrounding seamount in metres (KW2011)
sm_basin = extract_var("sm_basin")  # oceanic basin: 1=Atlantic, 2=Pacific, 3=Southern Ocean, 4=Indian Ocean
sm_height = extract_var("sm_height")  # seamount height in metres (KW2011)
sm_maj_axis = extract_var("sm_maj_axis")  # (a) major axis of basal ellipse in metres (KW2011)
sm_min_axis = extract_var("sm_min_axis")  # (b) minor axis of basal ellipse in metres (KW2011)
sm_azimuth = extract_var("sm_azimuth")  # azimuth of basal ellipse in degrees (KW2011)
sm_L = extract_var("sm_L")  # basal approx radius calculated as 0.5*sqrt(ab), in metres (KW2011)
sm_f = extract_var("sm_f")  # absolute value of Coriolis parameter
sm_hh_depth = extract_var("sm_hh_depth")  # depth of seamount half-height (KW2011)
sm_N = extract_var("sm_N")  # stratification N in s^{-1}, calculated from WOCE at nearest vertical level and horizontal grid point to seamount half height
sm_KE_ts = seamount_data["sm_KE_ts"]  # time series of KE from LLC4320 at two-week intervals at days "days_ts". For one day, velocities are decided with a 3 hourly average before KE is found and spatially averaged over a disk of radius sm_L + 10km at vertical level nearest to seamount half height, in m^2/s^2. This is performed for one day every 2 weeks for a year.
days_ts = extract_var("days_ts")  # days after 1 Dec 2011 of KE time series from LLC4320
sm_vel = extract_var("sm_vel")  # mean velocity at seamount, calculated as the square root of the mean of sm_KE_ts, in m/s
sm_Ro = extract_var("sm_Ro")  # Rossby number calculated as sm_vel/sm_f/sm_L
sm_Fr_depth = extract_var("sm_Fr_depth")  # Froude number calculated as sm_vel/sm_N/sm_depth
sm_Fr_height = extract_var("sm_Fr_height")  # Froude number calculated as sm_vel/sm_N/sm_height
sm_Bu_depth = extract_var("sm_Bu_depth")  # Burger number calculated as (sm_N*sm_depth/sm_f/sm_L)^2
sm_Bu_height = extract_var("sm_Bu_height")  # Burger number calculated as (sm_N*sm_height/sm_f/sm_L)^2
sm_vel_50m_Atl = extract_var("sm_vel_50m_Atl")  # Seamount velocity from Jonathan Gula"s Atlantic simulations, 50m above bottom and interpolated onto seamount location, in m/s
sm_vel_bot_Atl = extract_var("sm_vel_bot_Atl")  # As above in the bottom grid cell, in m/s

print(f"Successfully loaded seamount data with {len(sm_lon)} seamounts")

# Display basic statistics
print("\n" + "="*70)
print("SEAMOUNT DATASET SUMMARY")
print("="*70)
print(f"Total number of seamounts: {len(sm_lon)}")
print()

# Geographic distribution
print("Geographic Distribution:")
basin_names = ["Atlantic", "Pacific", "Southern Ocean", "Indian Ocean"]
for i, basin_name in enumerate(basin_names, 1):
    count = np.sum(sm_basin == i)
    percentage = count / len(sm_basin) * 100
    print(f"  {basin_name}: {count} seamounts ({percentage:.1f}%)")
print()
print("="*70)

# Create a pandas DataFrame for easier data manipulation
print("\nCreating pandas DataFrame...")

# Extract IDs properly
ids = []
for i in range(len(sm_lon)):
    if hasattr(sm_id[0, i], "item"):
        ids.append(sm_id[0, i].item() if hasattr(sm_id[0, i], "item") else str(sm_id[0, i]))
    else:
        ids.append(str(sm_id[0, i]))

df = pd.DataFrame({
    "id": ids,
    "longitude": sm_lon,
    "latitude": sm_lat,
    # "depth": sm_depth,
    "basin": sm_basin,
    "height": sm_height,
    "major_axis": sm_maj_axis,
    "minor_axis": sm_min_axis,
    # "azimuth": sm_azimuth,
    "basal_radius_L": sm_L,
    "coriolis_f": sm_f,
    # "half_height_depth": sm_hh_depth,
    "stratification_N": sm_N,
    "velocity": sm_vel,
    "rossby_number": sm_Ro,
    # "froude_depth": sm_Fr_depth,
    "froude_height": sm_Fr_height,
    # "burger_depth": sm_Bu_depth,
    "burger_height": sm_Bu_height,
    # "vel_50m_atlantic": sm_vel_50m_Atl,
    # "vel_bottom_atlantic": sm_vel_bot_Atl
})

# Add basin names
basin_map = {1: "Atlantic", 2: "Pacific", 3: "Southern Ocean", 4: "Indian Ocean"}
df["basin_name"] = df["basin"].map(basin_map)

print(f"DataFrame created with shape: {df.shape}")
print("\nDataFrame info:")
print(df.info())

# Save to NetCDF (around 3.8 MB)
df.to_xarray().to_netcdf("seamount_data.nc")
so = df[df.basin_name == "Southern Ocean"]