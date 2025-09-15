import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

plt.rcParams["figure.constrained_layout.use"] = True

seamounts = xr.open_dataset("../bathymetry/seamount_data.nc")
seamounts["Slope_Bu"] = seamounts.rossby_number / seamounts.froude_height
seamounts["velocity_cubed"] = seamounts.velocity**3
seamounts["dissip_height"] = seamounts.velocity**3 * seamounts.Slope_Bu / seamounts.height
seamounts["dissip_width"] = seamounts.velocity**3 * seamounts.Slope_Bu / seamounts.basal_radius_L

seamounts["total_volume"] = seamounts.basal_radius_L**2 * seamounts.height
seamounts["total_dissip_height"] = seamounts.dissip_height * seamounts.total_volume
seamounts["total_dissip_width"] = seamounts.dissip_width * seamounts.total_volume

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 10), sharex=True, sharey=True,
                         subplot_kw={'projection': ccrs.PlateCarree()})

fixed_options = dict(x="longitude", y="latitude", edgecolors="face", s=1, transform=ccrs.PlateCarree())

# Add land features to all subplots
for ax in axes.flat:
    ax.add_feature(cfeature.LAND, color='black', zorder=0)
    ax.add_feature(cfeature.COASTLINE, color='gray', linewidth=0.5, zorder=1)
    ax.set_global()

seamounts.plot.scatter(ax=axes[0, 0], hue="Slope_Bu",           **fixed_options, norm=LogNorm(), vmin=0.1, vmax=10)
seamounts.plot.scatter(ax=axes[0, 1], hue="velocity_cubed",     **fixed_options, norm=LogNorm(), vmin=1e-5, vmax=1e-2)

seamounts.plot.scatter(ax=axes[1, 0], hue="dissip_height", **fixed_options, norm=LogNorm(), vmin=1e-8, vmax=1e-5)
seamounts.plot.scatter(ax=axes[1, 1], hue="dissip_width",  **fixed_options, norm=LogNorm(), vmin=1e-9, vmax=1e-6)

seamounts.plot.scatter(ax=axes[2, 0], hue="total_dissip_height", **fixed_options, norm=LogNorm(), vmin=1e3, vmax=1e6)
seamounts.plot.scatter(ax=axes[2, 1], hue="total_dissip_width",  **fixed_options, norm=LogNorm(), vmin=1e2, vmax=1e5)