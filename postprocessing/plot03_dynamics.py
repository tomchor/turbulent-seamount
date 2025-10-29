import numpy as np
from src.aux02_plotting import letterize
import xarray as xr
from matplotlib import pyplot as plt
import pynanigans as pn
from matplotlib.colors import LogNorm
from cmocean import cm
from src.aux00_utils import open_simulation

# plt.rcParams["figure.constrained_layout.use"] = True

#+++ Load datasets
print("Reading datasets...")
simdata_path = "../simulations/data/"
postproc_path = "../postprocessing/data/"

resolution = "dz1"
buffer = 5
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
xyzi_L00 = open_simulation(simdata_path  + f"xyzi.balanus_Ro_b0.1_Fr_b1_L0_FWHM500_{resolution}.nc", **snap_opts)[["PV", "Ro", "Δz_aac"]]
xyzd_L00 = open_simulation(postproc_path + f"xyzd.balanus_Ro_b0.1_Fr_b1_L0_FWHM500_{resolution}.nc", **avgd_opts)
aaad_L00 = open_simulation(postproc_path + f"aaad.balanus_Ro_b0.1_Fr_b1_L0_FWHM500_{resolution}.nc", **avgd_opts).sel(buffer=10)
ds_L00 = xr.merge([xyzi_L00, xyzd_L00, aaad_L00])

xyzi_L08 = open_simulation(simdata_path  + f"xyzi.balanus_Ro_b0.1_Fr_b1_L0.8_FWHM500_{resolution}.nc", **snap_opts)[["PV", "Ro", "Δz_aac"]]
xyzd_L08 = open_simulation(postproc_path + f"xyzd.balanus_Ro_b0.1_Fr_b1_L0.8_FWHM500_{resolution}.nc", **avgd_opts)
aaad_L08 = open_simulation(postproc_path + f"aaad.balanus_Ro_b0.1_Fr_b1_L0.8_FWHM500_{resolution}.nc", **avgd_opts).sel(buffer=10)
ds_L08 = xr.merge([xyzi_L08, xyzd_L08, aaad_L08])
#---

#+++ Create new variables and restrict volume
def prepare_ds(ds,
               x_slice = slice(-1.5*ds_L00.FWHM, np.inf),
               z_slice = slice(0, ds_L00.Lz - ds_L00.h_sponge),
               t_slice = 20):
    print("  Restricting domain and selecting time...")
    # Restrict domain first and select time immediately to minimize data
    ds = ds.sel(z_aac=z_slice, x_caa=x_slice).sel(time=t_slice, method="nearest")

    print("  Calculating shear production components...")
    ds["⟨VSPR⟩ᶻ"] = ds["⟨SPR⟩ᶻ"].sel(j=3)
    ds["⟨HSPR⟩ᶻ"] = ds["⟨SPR⟩ᶻ"].sel(j=[1, 2]).sum("j")
    ds["⟨TSPR⟩ᶻ"] = ds["⟨SPR⟩ᶻ"].sum("j")

    return ds

print("Preparing L=0 dataset...")
ds_L00 = prepare_ds(ds_L00)
print("Preparing L=0.8 dataset...")
ds_L08 = prepare_ds(ds_L08)
print("Data preparation complete!")
#---

#+++ Create subplot grid
print("Creating subplot grid")
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 12), sharex=True, layout=None)
plt.subplots_adjust(wspace=0.05, hspace=0)

datasets = [(ds_L00, "0"), (ds_L08, "0.8")]
yticks = [-500, 0, 500]

# Define row configurations
row_configs = [
    dict(var="PV", plot_opts={"z": ds_L00.H / 3, "method": "nearest"}, xyi=True, label="PV",
     cmap="RdBu_r", vmin=lambda ds: -1.5*ds.N2_inf*abs(ds.f_0), vmax=lambda ds: 1.5*ds.N2_inf*abs(ds.f_0)),
    dict(var="⟨R̄o⟩ᶻ", plot_opts={}, xyi=False, label="Ro",
     cmap="RdBu_r", vmin=-0.4, vmax=0.4),
    dict(var="⟨ε̄ₖ⟩ᶻ", plot_opts={}, xyi=False, label="⟨ε̄ₖ⟩ᶻ", white_text=True,
     cmap="inferno", norm=LogNorm(vmin=1e-11, vmax=1e-8)),
    dict(var="⟨ε̄ₚ⟩ᶻ", plot_opts={}, xyi=False, label="⟨ε̄ₚ⟩ᶻ", white_text=True,
     cmap="inferno", norm=LogNorm(vmin=1e-11, vmax=1e-8))
]
#---

#+++ Plot all variables
for row_idx, config in enumerate(row_configs):
    var_name = config["var"]
    print(f"Plotting {var_name}")

    for i, (ds, L_str) in enumerate(datasets):
        ax = axes[row_idx, i]

        # Get data
        if config["xyi"]:
            data = ds[var_name].pnsel(**config["plot_opts"])
        else:
            data = ds[var_name]

        # Prepare plot kwargs
        plot_kwargs = {
            "ax": ax, "x": "x_caa", "cmap": config["cmap"],
            "add_colorbar": False, "rasterized": True
        }
        if "norm" in config:
            plot_kwargs["norm"] = config["norm"]
        else:
            vmin = config["vmin"](ds) if callable(config["vmin"]) else config["vmin"]
            vmax = config["vmax"](ds) if callable(config["vmax"]) else config["vmax"]
            plot_kwargs["vmin"] = vmin
            plot_kwargs["vmax"] = vmax

        im = data.plot.imshow(**plot_kwargs)

        # Set labels
        ax.set_title(f"L/FWHM = {L_str}" if row_idx == 0 else "")
        ax.set_xlabel("x [m]" if row_idx == len(row_configs)-1 else "")
        ax.set_yticks(yticks)
        ax.set_ylabel("y [m]" if i == 0 else "")
        if i > 0:
            ax.set_yticklabels([])
        ax.set_aspect('equal')

    # Add colorbar for the right panel
    cax = axes[row_idx, 1].inset_axes([0.8, 0.1, 0.03, 0.8],
                                       transform=axes[row_idx, 1].transAxes, clip_on=False)
    cbar = plt.colorbar(im, cax=cax, orientation="vertical", label=config["label"])

    # Set white text for certain colorbars
    if config.get("white_text", False):
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
        cbar.ax.yaxis.label.set_color('white')
#---

#+++ Save
letterize(axes.flatten(), x=0.05, y=0.9, fontsize=9)
print("Saving figure...")
fig.savefig(f"../figures/dynamics_comparison_{resolution}.png", dpi=300, bbox_inches="tight", pad_inches=0)
print("Done!")
#---