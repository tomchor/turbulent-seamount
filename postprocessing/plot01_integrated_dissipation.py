import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

plt.rcParams["figure.constrained_layout.use"] = True

seamounts = xr.open_dataset("../bathymetry/seamount_data.nc")
seamounts["Slope_Bu"] = seamounts.rossby_number / seamounts.froude_height
seamounts["dissip_height"] = seamounts.velocity**3 * seamounts.Slope_Bu / seamounts.height

seamounts["total_volume"] = seamounts.basal_radius_L**2 * seamounts.height
seamounts["total_dissip_height"] = seamounts.dissip_height * seamounts.total_volume

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharey=True,
                         gridspec_kw = dict(width_ratios = [3, 1]))

#+++ Plot scatter plot on the left axis
fixed_options = dict(x="longitude", y="latitude", edgecolors="face", s=1, transform=ccrs.PlateCarree())

# Add land features to all subplots
ax = axes[0]
ax.projection = ccrs.PlateCarree()
ax.add_feature(cfeature.LAND, color="black", zorder=0)
ax.add_feature(cfeature.COASTLINE, color="gray", linewidth=0.5, zorder=1)
ax.set_global()

im = seamounts.plot.scatter(ax=ax, hue="total_dissip_height", **fixed_options, norm=LogNorm(), vmin=1e3, vmax=1e6, add_colorbar=False)
cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1, shrink=0.8)
cbar.set_label("Total Dissipation")
#---

#+++ Plot integrated line plot on the right axis
# Bin seamounts data by latitude and longitude with 1 degree resolution
lat_bins = np.arange(-90, 91, 1)

# Create binned statistics for various quantities
ax = axes[1]
print("Binning data")
binned_seamounts = seamounts.groupby_bins("latitude", lat_bins).sum()
binned_seamounts.total_dissip_height.plot(ax=ax, y="latitude_bins")
ax.set_ylabel("Latitude")
ax.set_xlabel("Integrated dissipation")
#---
