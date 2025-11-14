import numpy as np
from src.aux02_plotting import letterize
import xarray as xr
from matplotlib import pyplot as plt
import pynanigans as pn
from matplotlib.colors import LogNorm
from cmocean import cm
from src.aux00_utils import open_simulation

#+++ Load datasets
print("Reading datasets...")
simdata_path = "../simulations/data/"
postproc_path = "../postprocessing/data/"

# Simulation parameters
Ro_b = 0.1
Fr_b = 1
L_rough = 0
L_smooth = 0.8
buffer = 5
resolution = 1
variables_xz = ["∫⁵εₖdy", "∫⁵εₚdy", "∫¹⁰εₖdy", "∫¹⁰εₚdy"]
variables_xy = ["⟨ε̄ₖ⟩ᶻ", "⟨ε̄ₚ⟩ᶻ"]

snap_opts = dict(use_advective_periods=True,
                 unique_times=True,
                 squeeze=True,
                 load=False,
                 get_grid=False,
                 open_dataset_kwargs=dict(chunks="auto"))
avgd_opts = dict(unique_times=False,
                 load=False,
                 get_grid=False,
                 open_dataset_kwargs=dict(chunks="auto"))

# Load rough topography datasets
print("Loading rough topography datasets...")
xyza_rough = open_simulation(postproc_path + f"xyza.balanus_Ro_b{Ro_b}_Fr_b{Fr_b}_L{L_rough}_FWHM500_dz{resolution}.nc", **avgd_opts)
aaad_rough = open_simulation(postproc_path + f"aaad.balanus_Ro_b{Ro_b}_Fr_b{Fr_b}_L{L_rough}_FWHM500_dz{resolution}.nc", **avgd_opts).sel(buffer=buffer)
ds_rough = xr.merge([xyza_rough[variables_xz], aaad_rough[variables_xy]])

# Load smooth topography datasets
print("Loading smooth topography datasets...")
xyza_smooth = open_simulation(postproc_path + f"xyza.balanus_Ro_b{Ro_b}_Fr_b{Fr_b}_L{L_smooth}_FWHM500_dz{resolution}.nc", **avgd_opts)
aaad_smooth = open_simulation(postproc_path + f"aaad.balanus_Ro_b{Ro_b}_Fr_b{Fr_b}_L{L_smooth}_FWHM500_dz{resolution}.nc", **avgd_opts).sel(buffer=buffer)
ds_smooth = xr.merge([xyza_smooth[variables_xz], aaad_smooth[variables_xy]])
#---

#+++ Create new variables and restrict volume
def prepare_ds(ds,
               x_slice = slice(-1.5*ds_rough.FWHM, np.inf),
               z_slice = slice(buffer, 1.3*ds_smooth.H),
               t_slice = 20):
    print("  Restricting domain and selecting time...")
    # Restrict domain first
    ds = ds.sel(z_aac=z_slice, x_caa=x_slice)

    # Select time - use last time for integrated dissipation, method="nearest" for time-averaged
    if "time" in ds.dims:
        # For variables with time dimension
        times = ds.time.values
        n_final = len(times) - 1

        # Select final time for integrated dissipation (from xyzi)
        eps_vars = [v for v in ds.data_vars if "∫" in v and "ε" in v]
        if eps_vars:
            ds_eps = ds[eps_vars].isel(time=n_final)
            ds_other = ds.drop_vars(eps_vars).sel(time=t_slice, method="nearest")
            ds = xr.merge([ds_eps, ds_other])
        else:
            ds = ds.sel(time=t_slice, method="nearest")

    ds["∫⁵εₖdy_normalized"] = ds["∫⁵εₖdy"] / ds.FWHM
    ds["∫⁵εₚdy_normalized"] = ds["∫⁵εₚdy"] / ds.FWHM
    ds["∫¹⁰εₖdy_normalized"] = ds["∫¹⁰εₖdy"] / ds.FWHM
    ds["∫¹⁰εₚdy_normalized"] = ds["∫¹⁰εₚdy"] / ds.FWHM

    return ds

print(f"Preparing L={L_rough} (rough) dataset...")
ds_rough = prepare_ds(ds_rough)
print(f"Preparing L={L_smooth} (smooth) dataset...")
ds_smooth = prepare_ds(ds_smooth)
print("Data preparation complete!")
#---

#+++ Create subplot grid
print("Creating subplot grid")
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(11, 10), sharex="col", sharey="row", constrained_layout=False)
plt.subplots_adjust(wspace=0.05, hspace=0.02)

datasets = [(ds_rough, str(L_rough)), (ds_smooth, str(L_smooth))]
yticks = [-500, 0, 500]

# Map buffer to superscript
integration_bound = "⁵" if buffer == 5 else "¹⁰"

# Define row configurations
row_configs = [
    dict(var=f"∫{integration_bound}εₖdy_normalized", plot_opts={}, xyi=False, label=f"∫ε̄ₖdy / W [m²/s³]", white_text=True, cmap="inferno", norm=LogNorm(vmin=2e-10, vmax=1e-7), plot_type="xz"),
    dict(var="⟨ε̄ₖ⟩ᶻ", plot_opts={}, xyi=False, label="⟨ε̄ₖ⟩ᶻ", white_text=True, cmap="inferno", norm=LogNorm(vmin=1e-10, vmax=1e-8), plot_type="xy"),
    dict(var=f"∫{integration_bound}εₚdy_normalized", plot_opts={}, xyi=False, label=f"∫ε̄ₚdy / W [m²/s³]", white_text=True, cmap="inferno", norm=LogNorm(vmin=1e-11, vmax=2e-9), plot_type="xz"),
    dict(var="⟨ε̄ₚ⟩ᶻ", plot_opts={}, xyi=False, label="⟨ε̄ₚ⟩ᶻ", white_text=True, cmap="inferno", norm=LogNorm(vmin=1e-11, vmax=1e-9), plot_type="xy")
]
#---

#+++ Plot all variables
for row_idx, config in enumerate(row_configs):
    var_name = config["var"]
    print(f"Plotting {var_name}")

    for i, (ds, L_str) in enumerate(datasets):
        ax = axes[row_idx, i]

        # Get data
        data = ds[var_name]

        # Prepare plot kwargs based on plot type
        plot_kwargs = dict(ax=ax, x="x_caa", cmap=config["cmap"],
                           add_colorbar=False, rasterized=True)
        if "norm" in config:
            plot_kwargs |= dict(norm=config["norm"])
        else:
            vmin = config["vmin"](ds) if callable(config["vmin"]) else config["vmin"]
            vmax = config["vmax"](ds) if callable(config["vmax"]) else config["vmax"]
            plot_kwargs |= dict(vmin=vmin, vmax=vmax)

        # Choose the right y-coordinate based on plot type
        if config["plot_type"] == "xy":
            # xy plot (plan view)
            im = data.plot.imshow(**plot_kwargs)
            ylabel = "y [m]"
            ax.set_yticks(yticks)
            ax.set_aspect(1)
        else:
            # xz plot (vertical slice)
            plot_kwargs["y"] = "z_aac"
            im = data.plot(**plot_kwargs)
            ylabel = "z [m]"
            ax.set_aspect(8)

        # Set labels
        ax.set_title(f"$L/W = {L_str}$" if row_idx == 0 else "")
        ax.set_xlabel("x [m]" if row_idx == len(row_configs)-1 else "")
        ax.set_ylabel(ylabel if i == 0 else "")

    # Add colorbar for the right panel
    cax = axes[row_idx, 1].inset_axes([0.75, 0.1, 0.03, 0.8],
                                       transform=axes[row_idx, 1].transAxes, clip_on=False)
    cbar = plt.colorbar(im, cax=cax, orientation="vertical", label=config["label"])

    if config.get("white_text", False):
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
        cbar.ax.yaxis.label.set_color("white")
        cbar.outline.set_edgecolor("white")
#---

#+++ Save
letterize(axes.flatten(), x=0.05, y=0.9, fontsize=9)
print("Saving figure...")
fig.savefig(f"../figures/dynamics_comparison_buffer{buffer}m_dz{resolution}.pdf", dpi=300, bbox_inches="tight", pad_inches=0)
print("Done!")
#---