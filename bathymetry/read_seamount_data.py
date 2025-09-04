# This script reads the seamount_data.mat file, which can be downloaded from
# https://zenodo.org/records/11145852/files/seamount_data.mat?download=1
# and which is part of the data made available by Mashayek et al. (2024):
# https://doi.org/10.1073/pnas.2322163121
import numpy as np
from scipy.io import loadmat
import pandas as pd

# Read the MATLAB file
print("Reading seamount_data.mat file...")
seamount_data = loadmat("../bathymetry/seamount_data.mat")

# Display available fields (excluding MATLAB metadata)
print("Available fields in seamount_data.mat:")
for key in sorted(seamount_data.keys()):
    if not key.startswith('__'):  # Skip MATLAB metadata
        data = seamount_data[key]
        if isinstance(data, np.ndarray):
            print(f"  {key}: {type(data)} of shape {data.shape}")
        else:
            print(f"  {key}: {type(data)}")

# Extract key variables (flatten 2D arrays if needed)
def extract_var(name):
    var = seamount_data[name]
    return var.flatten() if var.ndim > 1 else var

sm_id = seamount_data["sm_id"]  # Keep as is for string array
sm_lon = extract_var("sm_lon")
sm_lat = extract_var("sm_lat")
sm_depth = extract_var("sm_depth")
sm_basin = extract_var("sm_basin")
sm_height = extract_var("sm_height")
sm_maj_axis = extract_var("sm_maj_axis")
sm_min_axis = extract_var("sm_min_axis")
sm_azimuth = extract_var("sm_azimuth")
sm_L = extract_var("sm_L")
sm_f = extract_var("sm_f")
sm_hh_depth = extract_var("sm_hh_depth")
sm_N = extract_var("sm_N")
sm_KE_ts = seamount_data["sm_KE_ts"]
days_ts = extract_var("days_ts")
sm_vel = extract_var("sm_vel")
sm_Ro = extract_var("sm_Ro")
sm_Fr_depth = extract_var("sm_Fr_depth")
sm_Fr_height = extract_var("sm_Fr_height")
sm_Bu_depth = extract_var("sm_Bu_depth")
sm_Bu_height = extract_var("sm_Bu_height")
sm_vel_50m_Atl = extract_var("sm_vel_50m_Atl")
sm_vel_bot_Atl = extract_var("sm_vel_bot_Atl")

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


# Atlantic simulation data availability
atl_50m_available = np.sum(~np.isnan(sm_vel_50m_Atl))
atl_bot_available = np.sum(~np.isnan(sm_vel_bot_Atl))
print("Atlantic Simulation Data:")
print(f"  Velocity at 50m above bottom: {atl_50m_available} seamounts have data")
print(f"  Velocity at bottom: {atl_bot_available} seamounts have data")

print("="*70)

# Create a pandas DataFrame for easier data manipulation
print("\nCreating pandas DataFrame...")

# Extract IDs properly
ids = []
for i in range(len(sm_lon)):
    if hasattr(sm_id[0, i], 'item'):
        ids.append(sm_id[0, i].item() if hasattr(sm_id[0, i], 'item') else str(sm_id[0, i]))
    else:
        ids.append(str(sm_id[0, i]))

df = pd.DataFrame({
    'id': ids,
    'longitude': sm_lon,
    'latitude': sm_lat,
    # 'depth': sm_depth,
    'basin': sm_basin,
    'height': sm_height,
    'major_axis': sm_maj_axis,
    'minor_axis': sm_min_axis,
    # 'azimuth': sm_azimuth,
    'basal_radius_L': sm_L,
    'coriolis_f': sm_f,
    'half_height_depth': sm_hh_depth,
    'stratification_N': sm_N,
    'velocity': sm_vel,
    'rossby_number': sm_Ro,
    # 'froude_depth': sm_Fr_depth,
    'froude_height': sm_Fr_height,
    'burger_depth': sm_Bu_depth,
    'burger_height': sm_Bu_height,
    # 'vel_50m_atlantic': sm_vel_50m_Atl,
    # 'vel_bottom_atlantic': sm_vel_bot_Atl
})

# Add basin names
basin_map = {1: 'Atlantic', 2: 'Pacific', 3: 'Southern Ocean', 4: 'Indian Ocean'}
df['basin_name'] = df['basin'].map(basin_map)

print(f"DataFrame created with shape: {df.shape}")
print("\nDataFrame info:")
print(df.info())

print("\nFirst few rows of DataFrame:")
print(df.head())