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

fig = plt.figure(figsize=(10, 13))
gs = fig.add_gridspec(4, 2, width_ratios=[3, 1])

# Create left subplots with projection for heatmaps (4 rows)
ax_map_dissip_smooth = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
ax_map_dissip_rough = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
ax_map_mixing_smooth = fig.add_subplot(gs[2, 0], projection=ccrs.PlateCarree())
ax_map_mixing_rough = fig.add_subplot(gs[3, 0], projection=ccrs.PlateCarree())

# Create right subplots for line plots (only 2 rows, span 2 rows each)
ax_plot_dissip = fig.add_subplot(gs[0:2, 1])
ax_plot_mixing = fig.add_subplot(gs[2:4, 1])

#+++ Plot heatmaps on the left column (4 rows)
fixed_options = dict(x="longitude", y="latitude", edgecolors="face", s=1, transform=ccrs.PlateCarree())

# Row 1: KE dissipation for smooth seamounts
ax_map_dissip_smooth.add_feature(cfeature.LAND, color="black", zorder=0)
ax_map_dissip_smooth.add_feature(cfeature.COASTLINE, color="gray", linewidth=0.5, zorder=1)
ax_map_dissip_smooth.set_ylim(-80, 80)
ax_map_dissip_smooth.set_title("KE dissipation (smooth)")

gl1 = ax_map_dissip_smooth.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False)
gl1.top_labels = False
gl1.right_labels = False
gl1.bottom_labels = False

vmin_dissip = 1e1
vmax_dissip = 1e4
im_dissip_smooth = seamounts.plot.scatter(ax=ax_map_dissip_smooth, hue="tottal_dissip_smooth", cmap="YlOrRd", **fixed_options, norm=LogNorm(), vmin=vmin_dissip, vmax=vmax_dissip, add_colorbar=False)
cbar_dissip_smooth = plt.colorbar(im_dissip_smooth, ax=ax_map_dissip_smooth, orientation="vertical", pad=0.02, location="left", shrink=0.5)
cbar_dissip_smooth.set_label(r"Total KE dissipation [m$^2$/s$^3$ $\times$ m$^3$]")

# Row 2: KE dissipation for rough seamounts
ax_map_dissip_rough.add_feature(cfeature.LAND, color="black", zorder=0)
ax_map_dissip_rough.add_feature(cfeature.COASTLINE, color="gray", linewidth=0.5, zorder=1)
ax_map_dissip_rough.set_ylim(-80, 80)
ax_map_dissip_rough.set_title("KE dissipation (rough)")

gl2 = ax_map_dissip_rough.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False)
gl2.top_labels = False
gl2.right_labels = False
gl2.bottom_labels = False

im_dissip_rough = seamounts.plot.scatter(ax=ax_map_dissip_rough, hue="tottal_dissip_rough", cmap="YlOrRd", **fixed_options, norm=LogNorm(), vmin=vmin_dissip, vmax=vmax_dissip, add_colorbar=False)
cbar_dissip_rough = plt.colorbar(im_dissip_rough, ax=ax_map_dissip_rough, orientation="vertical", pad=0.02, location="left", shrink=0.5)
cbar_dissip_rough.set_label(r"Total KE dissipation [m$^2$/s$^3$ $\times$ m$^3$]")

# Row 3: PE mixing for smooth seamounts
ax_map_mixing_smooth.add_feature(cfeature.LAND, color="black", zorder=0)
ax_map_mixing_smooth.add_feature(cfeature.COASTLINE, color="gray", linewidth=0.5, zorder=1)
ax_map_mixing_smooth.set_ylim(-80, 80)
ax_map_mixing_smooth.set_title("PE mixing (linear)")

gl3 = ax_map_mixing_smooth.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False)
gl3.top_labels = False
gl3.right_labels = False
gl3.bottom_labels = False
labels = False

vmin_mixing = 1e1
vmax_mixing = 1e4
im_mixing_smooth = seamounts.plot.scatter(ax=ax_map_mixing_smooth, hue="total_mixing_linear", cmap="YlOrRd", **fixed_options, norm=LogNorm(), vmin=vmin_mixing, vmax=vmax_mixing, add_colorbar=False)
cbar_mixing_smooth = plt.colorbar(im_mixing_smooth, ax=ax_map_mixing_smooth, orientation="vertical", pad=0.02, location="left", shrink=0.5)
cbar_mixing_smooth.set_label(r"Total PE dissipation [m$^2$/s$^3$ $\times$ m$^3$]")

# Row 4: PE mixing for rough seamounts
ax_map_mixing_rough.add_feature(cfeature.LAND, color="black", zorder=0)
ax_map_mixing_rough.add_feature(cfeature.COASTLINE, color="gray", linewidth=0.5, zorder=1)
ax_map_mixing_rough.set_ylim(-80, 80)
ax_map_mixing_rough.set_title("PE mixing (quadratic)")

gl4 = ax_map_mixing_rough.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False)
gl4.top_labels = False
gl4.right_labels = False

im_mixing_rough = seamounts.plot.scatter(ax=ax_map_mixing_rough, hue="total_mixing_quadratic", cmap="YlOrRd", **fixed_options, norm=LogNorm(), vmin=vmin_mixing, vmax=vmax_mixing, add_colorbar=False)
cbar_mixing_rough = plt.colorbar(im_mixing_rough, ax=ax_map_mixing_rough, orientation="vertical", pad=0.02, location="left", shrink=0.5)
cbar_mixing_rough.set_label(r"Total PE dissipation [m$^2$/s$^3$ $\times$ m$^3$]")
#---

#+++ Plot integrated line plots on the right column (2 rows spanning)
# Bin seamounts data by latitude and longitude with 1 degree resolution
lat_bins = np.arange(-80, 81, 1)

# Create binned statistics for various quantities
print("Binning data")
binned_seamounts = seamounts.groupby_bins("latitude", lat_bins).sum()

# Top right: KE dissipation line plot
binned_seamounts.tottal_dissip_smooth.plot(ax=ax_plot_dissip, y="latitude_bins", color="blue", label="Smooth")
binned_seamounts.tottal_dissip_rough.plot(ax=ax_plot_dissip, y="latitude_bins", color="red", linestyle="--", label="Rough")
ax_plot_dissip.set_ylabel("Latitude (degrees)")
ax_plot_dissip.set_xlabel("Integrated dissipation")
ax_plot_dissip.set_title("Zonally-integrated\nKE dissipation")
ax_plot_dissip.legend()

# Bottom right: PE mixing line plot
binned_seamounts.total_mixing_linear.plot(ax=ax_plot_mixing, y="latitude_bins", color="blue", label="Linear")
binned_seamounts.total_mixing_quadratic.plot(ax=ax_plot_mixing, y="latitude_bins", color="red", linestyle="--", label="Quadratic")
ax_plot_mixing.set_ylabel("Latitude (degrees)")
ax_plot_mixing.set_xlabel("Integrated mixing")
ax_plot_mixing.set_title("Zonally-integrated\nPE mixing")
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
fig.savefig("../figures/global_dissipation.pdf", dpi=300, bbox_inches="tight", pad_inches=0)
#---