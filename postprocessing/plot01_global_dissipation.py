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

fig = plt.figure(figsize=(10, 5))
gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])

# Create left subplot with projection
ax_map = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())

# Create right subplot without projection, sharing y-axis with left subplot
ax_plot = fig.add_subplot(gs[1], sharey=ax_map)

#+++ Plot scatter plot on the left axis
fixed_options = dict(x="longitude", y="latitude", edgecolors="face", s=1, transform=ccrs.PlateCarree())

# Add land features to all subplots
ax_map.add_feature(cfeature.LAND, color="black", zorder=0)
ax_map.add_feature(cfeature.COASTLINE, color="gray", linewidth=0.5, zorder=1)
ax_map.set_ylim(-80, 80)
ax_map.set_title("Seamount dissipation")

# Add latitude and longitude gridlines and labels
gl = ax_map.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False)
gl.top_labels = False
gl.right_labels = False

im = seamounts.plot.scatter(ax=ax_map, hue="total_dissip_height", **fixed_options, norm=LogNorm(), vmin=1e3, vmax=5e5, add_colorbar=False)
cbar = plt.colorbar(im, ax=ax_map, orientation="horizontal", pad=-0.05, shrink=0.8, location="top")
cbar.set_label(r"Total KE dissipation [m$^2$/s$^3$ $\times$ m$^3$]")
#---

#+++ Plot integrated line plot on the right axis
# Bin seamounts data by latitude and longitude with 1 degree resolution
lat_bins = np.arange(-80, 81, 1)

# Create binned statistics for various quantities
print("Binning data")
binned_seamounts = seamounts.groupby_bins("latitude", lat_bins).sum()
binned_seamounts.total_dissip_height.plot(ax=ax_plot, y="latitude_bins", color="black")
ax_plot.set_ylabel("Latitude (degrees)")
ax_plot.set_xlabel("Integrated dissipation")
ax_plot.set_title("Zonally-integrated\nseamount dissipation")
#---

#+++ Save figure
fig.savefig("../figures/integrated_dissipation.png", dpi=300)
#---