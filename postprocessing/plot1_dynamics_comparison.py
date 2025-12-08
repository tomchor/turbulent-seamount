import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource, LogNorm
from src.aux00_utils import open_simulation, condense
from src.aux02_plotting import letterize

plt.rcParams["figure.constrained_layout.use"] = False

#+++ Parameters
simdata_path = "../simulations/data/"
postproc_path = "../postprocessing/data/"
simname_base = "balanus"
Ro_b = 0.1
Fr_b = 1
buffer = 5
resolution = 2
#---

#+++ Load datasets
dataset_opts = dict(use_advective_periods=True, unique_times=True, squeeze=True,
                    load=False, get_grid=False, open_dataset_kwargs=dict(chunks="auto"))
avgd_opts = dict(unique_times=False, load=False, get_grid=False,
                 open_dataset_kwargs=dict(chunks="auto"))

# Load data for both L values
L_values = ["0", "0.8"]
datasets = {}
dissipation_xz_datasets = {}
dissipation_xy_datasets = {}

for L_val in L_values:
    simname = f"{simname_base}_Ro_b{Ro_b}_Fr_b{Fr_b}_L{L_val}_dz{resolution}"

    xyzi = open_simulation(f"{simdata_path}xyzi.{simname}.nc", **dataset_opts)
    aaad = open_simulation(f"{postproc_path}aaad.{simname}.nc", **avgd_opts).sel(buffer=buffer)

    ds = xr.merge([xyzi, aaad], join="outer").sel(time=np.inf, method="nearest")
    datasets[L_val] = ds

    # Load dissipation data for ∫εₖdy (xz view)
    xyza = open_simulation(f"{postproc_path}xyza.{simname}.nc", **avgd_opts)
    xyza = condense(xyza, ["∫⁵εₖdy", "∫¹⁰εₖdy"], "∫εₖdy", dimname="buffer", indices=[5, 10])

    dissipation_ds = xyza[["∫εₖdy"]].sel(buffer=buffer)

    # Domain restriction
    full_width_half_maximum = dissipation_ds.FWHM
    dissipation_ds = dissipation_ds.sel(z_aac=slice(buffer, 1.3*dissipation_ds.H))

    # Time selection
    if "time" in dissipation_ds.dims:
        epsilon_variables = [variable_name for variable_name in dissipation_ds.data_vars if "∫" in variable_name]
        dataset_epsilon = dissipation_ds[epsilon_variables].isel(time=-1)
        dataset_other = dissipation_ds.drop_vars(epsilon_variables).sel(time=20, method="nearest")
        dissipation_ds = xr.merge([dataset_epsilon, dataset_other])

    # Normalize integrated dissipation
    dissipation_ds["∫εₖdy_normalized"] = dissipation_ds["∫εₖdy"] / full_width_half_maximum

    dissipation_xz_datasets[L_val] = dissipation_ds

    # Load dissipation data for ∫ε̄ₖdz (xy view)
    aaad_dissipation = open_simulation(f"{postproc_path}aaad.{simname}.nc", **avgd_opts)
    dissipation_xy_ds = aaad_dissipation[["∫ε̄ₖdz"]].sel(buffer=buffer)

    # Time selection
    if "time" in dissipation_xy_ds.dims:
        epsilon_variables = [variable_name for variable_name in dissipation_xy_ds.data_vars if "∫" in variable_name]
        dataset_epsilon = dissipation_xy_ds[epsilon_variables].isel(time=-1)
        dataset_other = dissipation_xy_ds.drop_vars(epsilon_variables).sel(time=20, method="nearest")
        dissipation_xy_ds = xr.merge([dataset_epsilon, dataset_other])

    # Normalize integrated dissipation by H
    dissipation_xy_ds["∫ε̄ₖdz_normalized"] = dissipation_xy_ds["∫ε̄ₖdz"] / dissipation_xy_ds.H

    dissipation_xy_datasets[L_val] = dissipation_xy_ds

# Extract parameters
H = datasets["0"].H
FWHM = datasets["0"].FWHM
bathymetry_extent = 1.3 * FWHM
xlims = (xyza.x_faa.min().values, xyza.x_faa.max().values)
#---

#+++ Create figure with manual plot area positioning
fig_width_inches = 14
fig_height_inches = 19
fig = plt.figure(figsize=(fig_width_inches, fig_height_inches))

xpanel_size_inches = 4.5
ypanel_size_inches = xpanel_size_inches * datasets["0"].Ly.item() / datasets["0"].Lx.item()
zpanel_size_inches = ypanel_size_inches * 14 * datasets["0"].Lz.item() / datasets["0"].Lx.item()

# Plot area positions: [left, bottom, width, height] in INCHES
# These define the actual plot area size, excluding tick labels, axis labels, and titles
# You can adjust these values to control the size and position of each plot area
plot_area_positions_inches = {
    # Row 0: 3D bathymetry plots
    "3d_0": [0.4, fig_height_inches - 3.5, 5.6, 3.84],  # Left 3D plot area [left, bottom, width, height] in inches
    "3d_1": [5.8, fig_height_inches - 3.5, 5.6, 3.84],  # Right 3D plot area
    
    # Row 1: PV plots
    "pv_0": [1.4, fig_height_inches - 5.5, xpanel_size_inches, ypanel_size_inches],  # Left PV plot area
    "pv_1": [6.3, fig_height_inches - 5.5, xpanel_size_inches, ypanel_size_inches],  # Right PV plot area
    
    # Row 2: Ro plots
    "ro_0": [1.4, fig_height_inches - 7.5, xpanel_size_inches, ypanel_size_inches],  # Left Ro plot area
    "ro_1": [6.3, fig_height_inches - 7.5, xpanel_size_inches, ypanel_size_inches],  # Right Ro plot area
    
    # Row 3: ∫εₖdy plots
    "eps_0": [1.4, fig_height_inches - 9.4, xpanel_size_inches, zpanel_size_inches],  # Left ∫εₖdy plot area
    "eps_1": [6.3, fig_height_inches - 9.4, xpanel_size_inches, zpanel_size_inches],  # Right ∫εₖdy plot area
    
    # Row 4: ∫ε̄ₖdz plots
    "epsdz_0": [1.4, fig_height_inches - 11.9, xpanel_size_inches, ypanel_size_inches],  # Left ∫ε̄ₖdz plot area
    "epsdz_1": [6.3, fig_height_inches - 11.9, xpanel_size_inches, ypanel_size_inches],  # Right ∫ε̄ₖdz plot area
}

# Convert inches to figure coordinates (0-1)
plot_area_positions = {}
for key, pos_inches in plot_area_positions_inches.items():
    left_inches, bottom_inches, width_inches, height_inches = pos_inches
    plot_area_positions[key] = [
        left_inches / fig_width_inches,           # left in figure coordinates
        bottom_inches / fig_height_inches,        # bottom in figure coordinates
        width_inches / fig_width_inches,          # width in figure coordinates
        height_inches / fig_height_inches         # height in figure coordinates
    ]

# Create axes with extra space for labels (will be adjusted after plotting)
# 3D axes for bathymetry
ax_3d = [
    fig.add_axes([0.05, 0.70, 0.42, 0.25], projection="3d"),
    fig.add_axes([0.53, 0.70, 0.42, 0.25], projection="3d")
]

# 2D axes for PV, Ro, ∫εₖdy, and ∫ε̄ₖdz
axes = np.array([
    [fig.add_axes([0.05, 0.50, 0.42, 0.18]),
     fig.add_axes([0.53, 0.50, 0.42, 0.18])],
    [fig.add_axes([0.05, 0.30, 0.42, 0.18]),
     fig.add_axes([0.53, 0.30, 0.42, 0.18])],
    [fig.add_axes([0.05, 0.05, 0.42, 0.23]),
     fig.add_axes([0.53, 0.05, 0.42, 0.23])],
    [fig.add_axes([0.05, 0.05, 0.42, 0.18]),
     fig.add_axes([0.53, 0.05, 0.42, 0.18])]
])
#---

#+++ Plot 3D bathymetry
ls = LightSource(azdeg=270, altdeg=45)

for i, L_val in enumerate(L_values):
    ds = datasets[L_val]
    bathy = ds["bottom_height"].pnsel(x=slice(-bathymetry_extent, +bathymetry_extent), y=slice(-bathymetry_extent, +bathymetry_extent))

    x, y = np.meshgrid(bathy.x.values, bathy.y.values)
    rgb = ls.shade(bathy.values, cmap=plt.cm.gist_earth, vert_exag=0.1, blend_mode="soft")

    ax_3d[i].plot_surface(x, y, bathy.values, rstride=2, cstride=2, facecolors=rgb,
                          linewidth=0, antialiased=False, shade=False, rasterized=True)
    ax_3d[i].set_xlabel("x [m]")
    ax_3d[i].set_ylabel("y [m]")
    ax_3d[i].set_zlabel("z [m]")
    ax_3d[i].view_init(elev=25, azim=135)
    ax_3d[i].set_box_aspect((1, 1, 0.3))
#---

#+++ Plot 2D fields
yticks = [-1000, -500, 0, 500, 1000]

# Determine common x-limits from both dissipation datasets
dissipation_xz_x_min = min(dissipation_xz_datasets["0"].x_caa.min().values, dissipation_xz_datasets["0.8"].x_caa.min().values)
dissipation_xz_x_max = max(dissipation_xz_datasets["0"].x_caa.max().values, dissipation_xz_datasets["0.8"].x_caa.max().values)

# Determine common x-limits for xy dissipation plots
dissipation_xy_x_min = min(dissipation_xy_datasets["0"].x_caa.min().values, dissipation_xy_datasets["0.8"].x_caa.min().values)
dissipation_xy_x_max = max(dissipation_xy_datasets["0"].x_caa.max().values, dissipation_xy_datasets["0.8"].x_caa.max().values)

# Row configurations
rows = [
    dict(var="PV", label="Potential vorticity", cmap="RdBu_r",
         get_data=lambda ds: ds["PV"].pnsel(z=H/3, method="nearest"),
         vmin=lambda ds: -1.5 * ds.N2_inf * abs(ds.f_0),
         vmax=lambda ds: 1.5 * ds.N2_inf * abs(ds.f_0),
         xlabel="", remove_xticks=True,
         data_source="datasets", plot_type="xy", aspect=None, yticks=yticks, ylabel="y [m]"),
    dict(var="⟨R̄o⟩ᶻ", label="⟨Ro⟩ᶻ", cmap="RdBu_r",
         get_data=lambda ds: ds["⟨R̄o⟩ᶻ"],
         vmin=lambda ds: -0.4,
         vmax=lambda ds: 0.4,
         xlabel="x [m]", remove_xticks=False,
         data_source="datasets", plot_type="xy", aspect=None, yticks=yticks, ylabel="y [m]"),
    dict(var="∫εₖdy_normalized", label="∫εₖdy / W [m²/s³]", cmap="inferno",
         get_data=lambda ds: ds["∫εₖdy_normalized"],
         vmin=None, vmax=None,
         norm=LogNorm(vmin=5e-10, vmax=5e-7),
         xlabel="x [m]", remove_xticks=True,
         data_source="dissipation_xz_datasets", plot_type="xz", aspect=None, yticks=None, ylabel="z [m]", white_colorbar=True),
    dict(var="∫ε̄ₖdz_normalized", label="∫ε̄ₖdz / H [m²/s³]", cmap="inferno",
         get_data=lambda ds: ds["∫ε̄ₖdz_normalized"],
         vmin=None, vmax=None,
         norm=LogNorm(vmin=5e-10, vmax=5e-7),
         xlabel="x [m]", remove_xticks=False,
         data_source="dissipation_xy_datasets", plot_type="xy", aspect=None, yticks=yticks, ylabel="y [m]", white_colorbar=True)
]

print("Plotting 2D fields...")
for row_idx, config in enumerate(rows):
    for col_idx, L_val in enumerate(L_values):
        # Select data source
        if config["data_source"] == "datasets":
            ds = datasets[L_val]
        elif config["data_source"] == "dissipation_xz_datasets":
            ds = dissipation_xz_datasets[L_val]
        else:  # dissipation_xy_datasets
            ds = dissipation_xy_datasets[L_val]
        
        ax = axes[row_idx, col_idx]

        # Get data
        if config["plot_type"] == "xy":
            data = config["get_data"](ds)
            vmin = config["vmin"](ds) if config["vmin"] is not None else None
            vmax = config["vmax"](ds) if config["vmax"] is not None else None
            
            im = data.plot.imshow(ax=ax, x="x_caa", cmap=config["cmap"],
                                  vmin=vmin, vmax=vmax, add_colorbar=False, rasterized=True)
        else:  # xz plot
            kwargs = dict(ax=ax, x="x_caa", y="z_aac", cmap=config["cmap"],
                          norm=config["norm"], add_colorbar=False, rasterized=True)
            im = ds[config["var"]].plot(**kwargs)

        # Add bathymetry mask for PV and ∫ε̄ₖdz plots
        if row_idx == 0:  # PV plot
            bathy_mask = datasets[L_val].peripheral_nodes_ccc.pnsel(z=H/3, method="nearest")
            bathy_mask.plot.imshow(ax=ax, cmap="Greys", vmin=0, vmax=1, origin="lower",
                                   alpha=0.25, zorder=2, add_colorbar=False)
        elif row_idx == 3:  # ∫ε̄ₖdz plot
            bathy_mask = datasets[L_val].peripheral_nodes_ccc.pnsel(z=H/3, method="nearest")
            bathy_mask.plot.imshow(ax=ax, cmap="Greys", vmin=0, vmax=1, origin="lower",
                                   alpha=0.25, zorder=2, add_colorbar=False)

        # Set labels and formatting
        ax.set_xlabel(config["xlabel"])
        ax.set_ylabel(config["ylabel"] if col_idx == 0 else "")
        ax.set_title("")
        
        if config["yticks"] is not None:
            ax.set_yticks(config["yticks"])
                
        ax.set_xlim(xlims[0], xlims[1])

        if config["remove_xticks"]:
            ax.set_xticklabels([])
        if col_idx > 0:
            ax.set_yticklabels([])

    # Add colorbar
    cax = axes[row_idx, 1].inset_axes([0.75, 0.1, 0.03, 0.8],
                                      transform=axes[row_idx, 1].transAxes, clip_on=False)
    cbar = plt.colorbar(im, cax=cax, orientation="vertical", label=config["label"])
    
    # White colorbar for dissipation plots
    if config.get("white_colorbar", False):
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
        cbar.ax.yaxis.label.set_color("white")
        cbar.outline.set_edgecolor("white")
#---

#+++ Set exact plot area positions (excluding labels)
# This sets the actual plot area size after all labels are added
for i in range(2):
    ax_3d[i].set_position(plot_area_positions[f"3d_{i}"])

for row_idx in range(4):
    for col_idx in range(2):
        if row_idx == 0:
            key = f"pv_{col_idx}"
        elif row_idx == 1:
            key = f"ro_{col_idx}"
        elif row_idx == 2:
            key = f"eps_{col_idx}"
        else:  # row_idx == 3
            key = f"epsdz_{col_idx}"
        axes[row_idx, col_idx].set_position(plot_area_positions[key])
#---

#+++ Finalize
delta = H.item() / FWHM.item()
fig.suptitle(f"Ro$_b$ = {datasets['0'].Ro_b.item()}, Fr$_b$ = {datasets['0'].Fr_b.item()}, S$_b$ = {datasets['0'].Slope_Bu.item()}, $\delta$ = {delta:.1f}",
             fontsize=14, y=0.995, x=0.43)
letterize(fig.axes[:10], x=0.05, y=0.75, fontsize=12,
          bbox=dict(boxstyle="square", facecolor="white", alpha=0.8))

output_path = f"../figures/{simname_base}_dynamics_comparison_L0_vs_L08_dz{resolution}_buffer{buffer}.pdf"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Saved plot to {output_path}")
#---
