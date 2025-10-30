import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

plt.rcParams["figure.constrained_layout.use"] = True

seamounts = xr.open_dataset("../bathymetry/seamount_data.nc")
seamounts["Slope_Bu"] = seamounts.rossby_number / seamounts.froude_height

#+++ Dissipation calculations
# Dissipation according to linear formula (smooth seamounts)
seamounts["dissip_linear"] = 1e-2 * seamounts.Slope_Bu * seamounts.velocity**3 / seamounts.height
seamounts["mixing_linear"] = 2e-2 * seamounts.Slope_Bu * seamounts.velocity**3 / seamounts.height
seamounts["mixing_quadratic"] = 2e-2 * seamounts.Slope_Bu**2 * seamounts.velocity**3 / seamounts.height

# Dissipation minimum value, which fixes a Slope_Bu but still depends on the velocity and height of seamount
Slope_Bu_threshold = 0.6
seamounts["dissip_minimum"] = 1e-2 * Slope_Bu_threshold * seamounts.velocity**3 / seamounts.height

# Put both together for piecewise dissipation
seamounts["dissip_piecewise"] = xr.where(seamounts.Slope_Bu > Slope_Bu_threshold,
                                         seamounts.dissip_linear,
                                         seamounts.dissip_minimum)

seamounts["total_volume"] = seamounts.basal_radius_L**2 * seamounts.height

seamounts["tottal_dissip_smooth"] = seamounts.dissip_linear * seamounts.total_volume
seamounts["tottal_dissip_rough"] = seamounts.dissip_piecewise * seamounts.total_volume

seamounts["total_mixing_linear"] = seamounts.mixing_linear * seamounts.total_volume
seamounts["total_mixing_quadratic"] = seamounts.mixing_quadratic * seamounts.total_volume
#---

fig = plt.figure(figsize=(10, 10))
gs = fig.add_gridspec(2, 2, width_ratios=[3, 1])

# Create left subplot with projection for dissipation (top row)
ax_map_dissip = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())

# Create right subplot without projection, sharing y-axis with left subplot
ax_plot_dissip = fig.add_subplot(gs[0, 1], sharey=ax_map_dissip)

# Create left subplot with projection for mixing (bottom row)
ax_map_mixing = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())

# Create right subplot without projection for mixing
ax_plot_mixing = fig.add_subplot(gs[1, 1], sharey=ax_map_mixing)

#+++ Plot dissipation scatter plot on the left axis (top row)
fixed_options = dict(x="longitude", y="latitude", edgecolors="face", s=1, transform=ccrs.PlateCarree())

# Add land features to dissipation map
ax_map_dissip.add_feature(cfeature.LAND, color="black", zorder=0)
ax_map_dissip.add_feature(cfeature.COASTLINE, color="gray", linewidth=0.5, zorder=1)
ax_map_dissip.set_ylim(-80, 80)
ax_map_dissip.set_title("Seamount dissipation")

# Add latitude and longitude gridlines and labels
gl = ax_map_dissip.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False)
gl.top_labels = False
gl.right_labels = False

im_dissip = seamounts.plot.scatter(ax=ax_map_dissip, hue="tottal_dissip_rough", cmap="YlOrRd", **fixed_options, norm=LogNorm(), vmin=1e1, vmax=1e3, add_colorbar=False)
cbar_dissip = plt.colorbar(im_dissip, ax=ax_map_dissip, orientation="vertical", pad=0.02, location="left", shrink=0.5)
cbar_dissip.set_label(r"Total KE dissipation [m$^2$/s$^3$ $\times$ m$^3$]")
#---

#+++ Plot integrated line plot on the right axis
# Bin seamounts data by latitude and longitude with 1 degree resolution
lat_bins = np.arange(-80, 81, 1)

# Create binned statistics for various quantities
print("Binning data")
binned_seamounts = seamounts.groupby_bins("latitude", lat_bins).sum()
binned_seamounts.tottal_dissip_smooth.plot(ax=ax_plot_dissip, y="latitude_bins", color="blue", label="Smooth")
binned_seamounts.tottal_dissip_rough.plot(ax=ax_plot_dissip, y="latitude_bins", color="red", linestyle="--", label="Rough")
ax_plot_dissip.set_ylabel("Latitude (degrees)")
ax_plot_dissip.set_xlabel("Integrated dissipation")
ax_plot_dissip.set_title("Zonally-integrated\nseamount dissipation")
ax_plot_dissip.legend()
#---

#+++ Plot mixing scatter plot on the left axis (bottom row)
# Add land features to mixing map
ax_map_mixing.add_feature(cfeature.LAND, color="black", zorder=0)
ax_map_mixing.add_feature(cfeature.COASTLINE, color="gray", linewidth=0.5, zorder=1)
ax_map_mixing.set_ylim(-80, 80)
ax_map_mixing.set_title("Seamount mixing")

# Add latitude and longitude gridlines and labels
gl_mix = ax_map_mixing.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False)
gl_mix.top_labels = False
gl_mix.right_labels = False

im_mixing = seamounts.plot.scatter(ax=ax_map_mixing, hue="total_mixing_quadratic", cmap="YlOrRd", **fixed_options, norm=LogNorm(), vmin=1e1, vmax=1e3, add_colorbar=False)
cbar_mixing = plt.colorbar(im_mixing, ax=ax_map_mixing, orientation="vertical", pad=0.02, location="left", shrink=0.5)
cbar_mixing.set_label(r"Total PE dissipation [m$^2$/s$^3$ $\times$ m$^3$]")
#---

#+++ Plot integrated mixing line plot on the right axis (bottom row)
binned_seamounts.total_mixing_linear.plot(ax=ax_plot_mixing, y="latitude_bins", color="blue", label="Linear")
binned_seamounts.total_mixing_quadratic.plot(ax=ax_plot_mixing, y="latitude_bins", color="red", linestyle="--", label="Quadratic")
ax_plot_mixing.set_ylabel("Latitude (degrees)")
ax_plot_mixing.set_xlabel("Integrated mixing")
ax_plot_mixing.set_title("Zonally-integrated\nseamount mixing")
ax_plot_mixing.legend()
#---

#+++ Calculate difference
global_ratio = seamounts.tottal_dissip_smooth.sum("index") / seamounts.tottal_dissip_rough.sum("index")

south = seamounts.where(seamounts.latitude<-40)
south_ratio = south.tottal_dissip_smooth.sum("index") / south.tottal_dissip_rough.sum("index")

print(f"Global ratio: {global_ratio:.2f}")
print(f"South ratio: {south_ratio:.2f}")
#---

#+++ Save figure
fig.savefig("../figures/integrated_dissipation.png", dpi=300)
#---