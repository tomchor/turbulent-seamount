import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from src.aux00_utils import open_simulation, condense
from src.aux02_plotting import letterize

#+++ Parameters
simname_base = "balanus"
simdata_path = "../simulations/data/"
postproc_path = "data/"
Ro_b = 0.1
Fr_b = 0.8
FWHM = 1000
Lx = 9000
Ly = 4000
L_rough = 0
L_smooth = 0.8
buffer = 10
resolution = 2
t_slice = 20
#---

#+++ Load datasets
print("Loading datasets...")
averaged_options = dict(unique_times=False, load=False, get_grid=False,
                        open_dataset_kwargs=dict(chunks="auto"))

# Variables to load
variables_xz = ["∫εₖdy", "∫εₚdy"]
variables_xy = ["∫ε̄ₖdz", "∫ε̄ₚdz"]

datasets = {}
for L_value, L_key in [(L_rough, "rough"), (L_smooth, "smooth")]:
    simulation_name = f"{simname_base}_Ro_b{Ro_b}_Fr_b{Fr_b}_L{L_value}_FWHM{FWHM}_Lx{Lx}_Ly{Ly}_dz{resolution}"

    xyza_dataset = open_simulation(f"{postproc_path}xyza.{simulation_name}.nc", **averaged_options)
    xyza_dataset = condense(xyza_dataset, ["∫⁵εₖdy", "∫¹⁰εₖdy"], "∫εₖdy", dimname="buffer", indices=[5, 10])
    xyza_dataset = condense(xyza_dataset, ["∫⁵εₚdy", "∫¹⁰εₚdy"], "∫εₚdy", dimname="buffer", indices=[5, 10])

    aaad_dataset = open_simulation(f"{postproc_path}aaad.{simulation_name}.nc", **averaged_options)

    dataset = xr.merge([xyza_dataset[variables_xz], aaad_dataset[variables_xy]]).sel(buffer=buffer)

    # Domain restriction
    full_width_half_maximum = dataset.FWHM
    seamount_height = dataset.H
    dataset = dataset.sel(z_aac=slice(buffer, 1.3*seamount_height), x_caa=slice(-1.5*full_width_half_maximum, np.inf))

    # Time selection
    if "time" in dataset.dims:
        epsilon_variables = [variable_name for variable_name in dataset.data_vars if "∫" in variable_name]
        dataset_epsilon = dataset[epsilon_variables].isel(time=-1)
        dataset_other = dataset.drop_vars(epsilon_variables).sel(time=t_slice, method="nearest")
        dataset = xr.merge([dataset_epsilon, dataset_other])

    # Normalize integrated dissipation
    for variable_name in variables_xz:
        dataset[f"{variable_name}_normalized"] = dataset[variable_name] / full_width_half_maximum
    for variable_name in variables_xy:
        dataset[f"{variable_name}_normalized"] = dataset[variable_name] / seamount_height

    datasets[L_key] = dataset

print("Data loaded!")
#---

#+++ Create figure and subplots
fig, axes = plt.subplots(ncols=2, nrows=4, figsize=(10, 9), gridspec_kw=dict(hspace=-0.05, wspace=0.05), sharex=True, sharey="row")
#---

#+++ Plot variables
rows = [
    dict(var="∫εₖdy_normalized", label="∫εₖdy / W [m²/s³]", norm=LogNorm(vmin=2e-10, vmax=1e-7), plot_type="xz"),
    dict(var="∫ε̄ₖdz_normalized", label="∫ε̄ₖdz / H [m²/s³]", norm=LogNorm(vmin=2e-10, vmax=1e-7), plot_type="xy"),
    dict(var="∫εₚdy_normalized", label="∫εₚdy / W [m²/s³]", norm=LogNorm(vmin=1e-11, vmax=2e-9), plot_type="xz"),
    dict(var="∫ε̄ₚdz_normalized", label="∫ε̄ₚdz / H [m²/s³]", norm=LogNorm(vmin=1e-11, vmax=2e-9), plot_type="xy"),
]

yticks = [-500, 0, 500]
L_vals = [("rough", str(L_rough)), ("smooth", str(L_smooth))]

# Determine common x-limits from both datasets
x_min = min(datasets["rough"].x_caa.min().values, datasets["smooth"].x_caa.min().values)
x_max = max(datasets["rough"].x_caa.max().values, datasets["smooth"].x_caa.max().values)

print("Plotting...")
for row_idx, config in enumerate(rows):
    for col_idx, (ds_name, L_str) in enumerate(L_vals):
        ax = axes[row_idx, col_idx]
        ds = datasets[ds_name]

        # Plot kwargs
        kwargs = dict(ax=ax, x="x_caa", cmap="inferno", norm=config["norm"],
                      add_colorbar=False, rasterized=True)

        if config["plot_type"] == "xy":
            im = ds[config["var"]].plot.imshow(**kwargs)
            ylabel = "y [m]"
            ax.set_yticks(yticks)
            ax.set_aspect(1)
        else:
            kwargs["y"] = "z_aac"
            im = ds[config["var"]].plot(**kwargs)
            ylabel = "z [m]"
            ax.set_aspect(24)

        # Set common x-limits for all plots
        ax.set_xlim(x_min, x_max)
        ax.set_title(f"$L/W = {L_str}$" if row_idx == 0 else "")
        ax.set_xlabel("x [m]" if row_idx == len(rows)-1 else "")
        ax.set_ylabel(ylabel if col_idx == 0 else "")

        # if col_idx > 0:
        #     ax.set_yticklabels([])
        # if row_idx == 0 or row_idx == 2:
        #     ax.set_xticklabels([])

    # Colorbar
    cax = axes[row_idx, 1].inset_axes([0.75, 0.1, 0.03, 0.8],
                                      transform=axes[row_idx, 1].transAxes, clip_on=False)
    cbar = plt.colorbar(im, cax=cax, orientation="vertical", label=config["label"])
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    cbar.ax.yaxis.label.set_color("white")
    cbar.outline.set_edgecolor("white")
#---

#+++ Save
letterize(axes.flatten(), x=0.05, y=0.9, fontsize=9)
fig.savefig(f"../figures/{simname_base}_flat_eps_comparison_buffer{buffer}m_dz{resolution}.pdf",
            dpi=300, bbox_inches="tight", pad_inches=0)
print("Done!")
#---

