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

# Regular balanus parameters (from run_all_simulations.py)
Ro_b = 0.1
Fr_b = 0.8

# Flat balanus parameters (from plotS1_eps_flat.py)
FWHM_flat = 1000
Lx_flat = 9000
Ly_flat = 4000

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

# Variables to load (only epsilon_k)
variables_xz = ["∫εₖdy"]
variables_xy = ["∫ε̄ₖdz"]

datasets = {}

# Load regular balanus datasets
for L_value, L_key in [(L_rough, "reg_L0"), (L_smooth, "reg_L08")]:
    simulation_name = f"{simname_base}_Ro_b{Ro_b}_Fr_b{Fr_b}_L{L_value}_dz{resolution}"

    xyza_dataset = open_simulation(f"{postproc_path}xyza.{simulation_name}.nc", **averaged_options)
    xyza_dataset = condense(xyza_dataset, ["∫⁵εₖdy", "∫¹⁰εₖdy"], "∫εₖdy", dimname="buffer", indices=[5, 10])

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

    # Normalize x and y coordinates by FWHM
    dataset["x_caa_normalized"] = dataset.x_caa / full_width_half_maximum
    dataset["y_aca_normalized"] = dataset.y_aca / full_width_half_maximum

    datasets[L_key] = dataset

# Load flat balanus datasets
for L_value, L_key in [(L_rough, "flat_L0"), (L_smooth, "flat_L08")]:
    simulation_name = f"{simname_base}_Ro_b{Ro_b}_Fr_b{Fr_b}_L{L_value}_FWHM{FWHM_flat}_Lx{Lx_flat}_Ly{Ly_flat}_dz{resolution}"

    xyza_dataset = open_simulation(f"{postproc_path}xyza.{simulation_name}.nc", **averaged_options)
    xyza_dataset = condense(xyza_dataset, ["∫⁵εₖdy", "∫¹⁰εₖdy"], "∫εₖdy", dimname="buffer", indices=[5, 10])

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

    # Normalize x and y coordinates by FWHM
    dataset["x_caa_normalized"] = dataset.x_caa / full_width_half_maximum
    dataset["y_aca_normalized"] = dataset.y_aca / full_width_half_maximum

    datasets[L_key] = dataset

print("Data loaded!")
#---

#+++ Create figure and subplots
fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(9, 5), gridspec_kw=dict(hspace=0.15, wspace=0.05), sharex=True, sharey="row")
#---

#+++ Plot variables
# Row 0: Regular balanus (L=0 left, L=0.8 right)
# Row 1: Flat balanus (L=0 left, L=0.8 right)
# All panels show xy view of epsilon_k (∫ε̄ₖdz)

plot_config = dict(var="∫ε̄ₖdz_normalized", label="∫ε̄ₖdz / H [m²/s³]", norm=LogNorm(vmin=1e-10, vmax=1e-7))

# Dataset keys: (row_idx, col_idx) -> dataset_key
dataset_map = {
    (0, 0): "reg_L0",   # Regular balanus, L=0
    (0, 1): "reg_L08",  # Regular balanus, L=0.8
    (1, 0): "flat_L0",  # Flat balanus, L=0
    (1, 1): "flat_L08", # Flat balanus, L=0.8
}

# Determine common normalized x-limits and y-limits from all datasets
x_min = min([ds.x_caa_normalized.min().values for ds in datasets.values()])
x_max = max([ds.x_caa_normalized.max().values for ds in datasets.values()])
y_min = min([ds.y_aca_normalized.min().values for ds in datasets.values()])
y_max = max([ds.y_aca_normalized.max().values for ds in datasets.values()])

print("Plotting...")
# Store delta values for each row
delta_values = {}

for row_idx in range(2):
    for col_idx in range(2):
        ax = axes[row_idx, col_idx]
        ds_key = dataset_map[(row_idx, col_idx)]
        ds = datasets[ds_key]

        # Calculate and store delta (H/W) from dataset
        delta = ds.H.item() / ds.FWHM.item()
        if col_idx == 0:
            delta_values[row_idx] = delta

        # Create a temporary dataset with normalized coordinates for plotting
        ds_plot = ds.assign_coords(x_caa=ds.x_caa_normalized, y_aca=ds.y_aca_normalized)

        # Plot xy view using normalized coordinates
        kwargs = dict(ax=ax, x="x_caa", cmap="inferno", norm=plot_config["norm"],
                      add_colorbar=False, rasterized=True)
        im = ds_plot[plot_config["var"]].plot.imshow(**kwargs)

        ax.set_aspect(1)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # Titles: L values for columns in top row
        if row_idx == 0:
            L_str = str(L_rough) if col_idx == 0 else str(L_smooth)
            ax.set_title(f"$L/W = {L_str}$")
        else:
            ax.set_title("")

        # Labels
        ax.set_xlabel("$x/W$" if row_idx == 1 else "")
        ax.set_ylabel("$y/W$" if col_idx == 0 else "")

    # Colorbar for each row
    cax = axes[row_idx, 1].inset_axes([0.75, 0.1, 0.03, 0.8],
                                      transform=axes[row_idx, 1].transAxes, clip_on=False)
    cbar = plt.colorbar(im, cax=cax, orientation="vertical", label=plot_config["label"])
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    cbar.ax.yaxis.label.set_color("white")
    cbar.outline.set_edgecolor("white")
#---

#+++ Save
letterize(axes.flatten(), x=0.05, y=0.9, fontsize=9)
fig.savefig(f"../figures/{simname_base}_eps_flat_comparison_buffer{buffer}m_dz{resolution}.pdf",
            dpi=300, bbox_inches="tight", pad_inches=0)
print("Done!")
#---

