import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from src.aux02_plotting import letterize

plt.rcParams["figure.constrained_layout.use"] = True

seamounts = xr.open_dataset("../bathymetry/seamount_data.nc")
seamounts["Slope_Bu"] = seamounts.rossby_number / seamounts.froude_height
ρ = 1025
top_lat = 75

#+++ Dissipation calculations
# Dissipation according to linear formula (smooth seamounts)
seamounts["dissip_linear"] = 2e-2 * seamounts.Slope_Bu
seamounts["dissip_minimum"] = 2e-2
seamounts["dissip_piecewise"] = np.maximum(seamounts.dissip_linear, seamounts.dissip_minimum)

seamounts["mixing_linear"] = 2e-2 * seamounts.Slope_Bu
seamounts["mixing_quadratic"] = 2e-2 * seamounts.Slope_Bu**2

seamounts["dissip_scale"] = seamounts.velocity**3 / seamounts.height
seamounts["mixing_scale"] = seamounts.dissip_scale
seamounts["total_volume"] = seamounts.basal_radius_L**2 * seamounts.height

seamounts["total_dissip_smooth"] = seamounts.dissip_linear * seamounts.dissip_scale * seamounts.total_volume
seamounts["total_dissip_rough"] = seamounts.dissip_piecewise * seamounts.dissip_scale * seamounts.total_volume

seamounts["total_mixing_linear"] = seamounts.mixing_linear * seamounts.mixing_scale * seamounts.total_volume
seamounts["total_mixing_quadratic"] = seamounts.mixing_quadratic * seamounts.mixing_scale * seamounts.total_volume
#---

#+++ Convert to watts
seamounts["total_dissip_smooth"] = seamounts.total_dissip_smooth * ρ
seamounts.total_mixing_linear.attrs = dict(units="W", longer_name="KE dissipation")

seamounts["total_dissip_rough"] = seamounts.total_dissip_rough * ρ
seamounts.total_mixing_linear.attrs = dict(units="W", longer_name="Buoyancy mixing")

seamounts["total_mixing_linear"] = seamounts.total_mixing_linear * ρ
seamounts.total_mixing_linear.attrs = dict(units="W", longer_name="Buoyancy mixing")

seamounts["total_mixing_quadratic"] = seamounts.total_mixing_quadratic * ρ
seamounts.total_mixing_quadratic.attrs = dict(units="W", longer_name="Buoyancy mixing")
#---

#+++ Plotting function
def plot_heatmaps_and_line(data, var_top, var_bottom, var_line1, var_line2,
                            title_top, title_bottom, colorbar_label, xlabel, line_title,
                            legend_labels, vmin, vmax, cmap, output_filename, top_lat):
    """Plot two heatmaps (top and bottom) and a line plot on the right."""
    # Create figure with projections
    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], hspace=0)

    # Create left subplots with projection for heatmaps (2 rows)
    ax_map_top = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax_map_bottom = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())

    # Create right subplot for line plot (spans 2 rows)
    ax_plot = fig.add_subplot(gs[0:2, 1])

    # Plot heatmaps on the left column (2 rows)
    fixed_options = dict(x="longitude", y="latitude", edgecolors="face", s=1,
                         transform=ccrs.PlateCarree(), rasterized=True)

    # Row 1: Top heatmap
    gl1 = ax_map_top.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False)
    gl1.top_labels = False
    gl1.right_labels = False
    gl1.bottom_labels = False

    im_top = data.plot.scatter(ax=ax_map_top, hue=var_top, cmap=cmap, **fixed_options,
                                norm=LogNorm(), vmin=vmin, vmax=vmax, add_colorbar=False)
    cbar_top = plt.colorbar(im_top, ax=ax_map_top, orientation="vertical", pad=0.02,
                            location="left", shrink=0.5)
    cbar_top.set_label(colorbar_label)
    ax_map_top.add_feature(cfeature.LAND, color="black", zorder=0)
    ax_map_top.add_feature(cfeature.COASTLINE, color="gray", linewidth=0.5, zorder=1)
    ax_map_top.set_ylim(-top_lat, top_lat)
    ax_map_top.set_title(title_top)

    # Row 2: Bottom heatmap
    gl2 = ax_map_bottom.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False)
    gl2.top_labels = False
    gl2.right_labels = False

    im_bottom = data.plot.scatter(ax=ax_map_bottom, hue=var_bottom, cmap=cmap, **fixed_options,
                                   norm=LogNorm(), vmin=vmin, vmax=vmax, add_colorbar=False)
    cbar_bottom = plt.colorbar(im_bottom, ax=ax_map_bottom, orientation="vertical", pad=0.02,
                                location="left", shrink=0.5)
    cbar_bottom.set_label(colorbar_label)
    ax_map_bottom.add_feature(cfeature.LAND, color="black", zorder=0)
    ax_map_bottom.add_feature(cfeature.COASTLINE, color="gray", linewidth=0.5, zorder=1)
    ax_map_bottom.set_ylim(-top_lat, top_lat)
    ax_map_bottom.set_title(title_bottom)

    # Plot integrated line plot on the right column (spanning 2 rows)
    lat_bins = np.arange(-top_lat, top_lat + 1, 1)
    print("Binning data")
    binned_data = data.groupby_bins("latitude", lat_bins).sum()

    binned_data[var_line1].plot(ax=ax_plot, y="latitude_bins", color="blue", linestyle="--",
                                 label=legend_labels[0])
    binned_data[var_line2].plot(ax=ax_plot, y="latitude_bins", color="red", linestyle="--",
                                 label=legend_labels[1])
    ax_plot.set_ylabel("Latitude (degrees)")
    ax_plot.set_xlabel(xlabel)
    ax_plot.set_ylim(-top_lat, top_lat)
    ax_plot.set_title(line_title)
    ax_plot.grid(True)
    ax_plot.legend(loc="upper right", borderaxespad=0, framealpha=0.9, edgecolor="black",
                   fancybox=False)

    # Save figure
    letterize(fig.axes[:3], x=0.05, y=0.92, fontsize=12,
              bbox=dict(boxstyle="square", facecolor="white", alpha=0.9))
    fig.savefig(output_filename, dpi=300, bbox_inches="tight", pad_inches=0)
#---

#+++ Plot dissipation maps
plot_heatmaps_and_line(
    data=seamounts,
    var_top="total_dissip_smooth",
    var_bottom="total_dissip_rough",
    var_line1="total_dissip_smooth",
    var_line2="total_dissip_rough",
    title_top="KE dissipation (smooth)",
    title_bottom="KE dissipation (rough)",
    colorbar_label=r"Dissipation [W]",
    xlabel="KE dissipation [W]",
    line_title="KE dissipation\nper degree of latitude",
    legend_labels=("Smooth", "Rough"),
    vmin=1e4,
    vmax=1e8,
    cmap="YlOrRd",
    output_filename="../figures/global_dissipation.pdf",
    top_lat=top_lat
)
#---

#+++ Plot mixing maps
plot_heatmaps_and_line(
    data=seamounts,
    var_top="total_mixing_linear",
    var_bottom="total_mixing_quadratic",
    var_line1="total_mixing_linear",
    var_line2="total_mixing_quadratic",
    title_top="Buoyancy mixing (linear)",
    title_bottom="Buoyancy mixing (quadratic)",
    colorbar_label=r"Buoyancy mixing [W]",
    xlabel="Buoyancy mixing [W]",
    line_title="Buoyancy mixing\nper degree of latitude",
    legend_labels=("Smooth", "Rough"),
    vmin=1e3,
    vmax=1e7,
    cmap="GnBu",
    output_filename="../figures/global_mixing.pdf",
    top_lat=top_lat
)
#---

#+++ Calculate difference
global_dissip_ratio = seamounts.total_dissip_smooth.sum("index") / seamounts.total_dissip_rough.sum("index")
global_mixing_ratio = seamounts.total_mixing_linear.sum("index") / seamounts.total_mixing_quadratic.sum("index")

south = seamounts.where(seamounts.latitude<-40)
south_dissip_ratio = south.total_dissip_smooth.sum("index") / south.total_dissip_rough.sum("index")
south_mixing_ratio = south.total_mixing_linear.sum("index") / south.total_mixing_quadratic.sum("index")

print(f"Global dissipation ratio: {global_dissip_ratio:.2f}")
print(f"South dissipation ratio: {south_dissip_ratio:.2f}")

print(f"Global mixing ratio: {global_mixing_ratio:.2f}")
print(f"South mixing ratio: {south_mixing_ratio:.2f}")
#---