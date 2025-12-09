import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource, LogNorm, NoNorm
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
resolution = 1
#---

#+++ Load datasets
dataset_opts = dict(use_advective_periods=True, unique_times=True, squeeze=True,
                    load=False, get_grid=False, open_dataset_kwargs=dict(chunks="auto"))
avgd_opts = dict(unique_times=False, load=False, get_grid=False,
                 open_dataset_kwargs=dict(chunks="auto"))

# Load and merge all data for both L values
L_values = ["0", "0.8"]
datasets = {}

for L_val in L_values:
    simname = f"{simname_base}_Ro_b{Ro_b}_Fr_b{Fr_b}_L{L_val}_dz{resolution}"

    # Load all datasets
    xyzi = open_simulation(f"{simdata_path}xyzi.{simname}.nc", **dataset_opts)
    aaad = open_simulation(f"{postproc_path}aaad.{simname}.nc", **avgd_opts).sel(buffer=buffer)
    xyza = open_simulation(f"{postproc_path}xyza.{simname}.nc", **avgd_opts)

    # Process xyzi: extract bottom height
    xyzi_ds = xyzi[["bottom_height", "PV", "peripheral_nodes_ccc"]].sel(z_aac=xyzi.H/3, method="nearest").reset_coords(drop=True)

    # Process xyza: condense and extract dissipation variables
    xyza = condense(xyza, ["∫⁵ε̄ₖdy", "∫¹⁰ε̄ₖdy"], "∫ε̄ₖdy", dimname="buffer", indices=[5, 10])
    xyza = condense(xyza, ["∫⁵ε̄ₚdy", "∫¹⁰ε̄ₚdy"], "∫ε̄ₚdy", dimname="buffer", indices=[5, 10])
    xyza_ds = xyza[["∫ε̄ₖdy", "∫ε̄ₚdy", "bottom_height"]].sel(buffer=buffer)
    xyza_ds = xyza_ds.sel(z_aac=slice(buffer, 1.3*xyza_ds.H))

    # Process aaad: extract ∫ε̄ₖdz
    aaad_ds = aaad[["∫ε̄ₖdz", "∫ε̄ₚdz", "⟨R̄o⟩ᶻ"]]

    # Time selection for dissipation variables
    if "time" in xyza_ds.dims:
        epsilon_vars = [v for v in xyza_ds.data_vars if "∫" in v]
        xyza_ds = xr.merge([
            xyza_ds[epsilon_vars].isel(time=-1),
            xyza_ds.drop_vars(epsilon_vars).sel(time=20, method="nearest")
        ])
    if "time" in aaad_ds.dims:
        epsilon_vars = [v for v in aaad_ds.data_vars if "∫" in v]
        aaad_ds = xr.merge([
            aaad_ds[epsilon_vars].isel(time=-1),
            aaad_ds.drop_vars(epsilon_vars).sel(time=20, method="nearest")
        ])

    # Normalize dissipation variables
    xyza_ds["∫ε̄ₖdy_normalized"] = xyza_ds["∫ε̄ₖdy"] / xyza_ds.FWHM
    aaad_ds["∫ε̄ₖdz_normalized"] = aaad_ds["∫ε̄ₖdz"] / aaad_ds.H
    xyza_ds["∫ε̄ₚdy_normalized"] = xyza_ds["∫ε̄ₚdy"] / xyza_ds.FWHM
    aaad_ds["∫ε̄ₚdz_normalized"] = aaad_ds["∫ε̄ₚdz"] / aaad_ds.H

    # Merge everything
    ds = xr.merge([xyzi_ds, xyza_ds, aaad_ds], join="outer", compat="no_conflicts")
    ds = ds.sel(z_aac=slice(buffer, 1.3*ds.H)).sel(time=np.inf, method="nearest")

    datasets[L_val] = ds

# Extract parameters
ds = datasets["0"]
H = ds.H
FWHM = ds.FWHM
bathymetry_extent = 1.3 * FWHM
xlims = (-ds.runway_length_fraction_FWHM * ds.FWHM, ds.x_caa.max().values)
f_0 = ds.f_0.item()
N2_inf = ds.N2_inf.item()
#---

#+++ Create figure with manual plot area positioning
fig_width_inches = 14
fig_height_inches = 23
fig = plt.figure(figsize=(fig_width_inches, fig_height_inches))

xpanel_size_inches = 4.5
ypanel_size_inches = xpanel_size_inches * datasets["0"].Ly.item() / datasets["0"].Lx.item()
zpanel_size_inches = ypanel_size_inches * 14 * datasets["0"].Lz.item() / datasets["0"].Lx.item()

# Plot area positions: [left, bottom, width, height] in INCHES
# These define the actual plot area size, excluding tick labels, axis labels, and titles
# You can adjust these values to control the size and position of each plot area
plot_area_positions_inches = {
    # Row 0: 3D bathymetry plots
    "3d_0": [0.4, fig_height_inches - 3.2, 5.6, 3.84],  # Left 3D plot area [left, bottom, width, height] in inches
    "3d_1": [5.8, fig_height_inches - 3.2, 5.6, 3.84],  # Right 3D plot area

    # Row 1: PV plots
    "pv_0": [1.4, fig_height_inches - 4.9,                                xpanel_size_inches, ypanel_size_inches],  # Left PV plot area
    "pv_1": [6.3, fig_height_inches - 4.9,                                xpanel_size_inches, ypanel_size_inches],  # Right PV plot area

    # Row 2: Ro plots
    "ro_0": [1.4, fig_height_inches - 6.9,                                xpanel_size_inches, ypanel_size_inches],  # Left Ro plot area
    "ro_1": [6.3, fig_height_inches - 6.9,                                xpanel_size_inches, ypanel_size_inches],  # Right Ro plot area

    # Row 3: ∫εₖdy plots
    "eps_0": [1.4, fig_height_inches - 8.5,                               xpanel_size_inches, zpanel_size_inches],  # Left ∫εₖdy plot area
    "eps_1": [6.3, fig_height_inches - 8.5,                               xpanel_size_inches, zpanel_size_inches],  # Right ∫εₖdy plot area

    # Row 4: ∫ε̄ₖdz plots
    "epsdz_0": [1.4, fig_height_inches - 8.5 - 1.65*zpanel_size_inches,   xpanel_size_inches, ypanel_size_inches],  # Left ∫ε̄ₖdz plot area
    "epsdz_1": [6.3, fig_height_inches - 8.5 - 1.65*zpanel_size_inches,   xpanel_size_inches, ypanel_size_inches],  # Right ∫ε̄ₖdz plot area

    # Row 5: ∫ε̄ₚdy plots
    "epspdy_0": [1.4, fig_height_inches - 12.2,                           xpanel_size_inches, zpanel_size_inches],  # Left ∫ε̄ₚdy plot area
    "epspdy_1": [6.3, fig_height_inches - 12.2,                           xpanel_size_inches, zpanel_size_inches],  # Right ∫ε̄ₚdy plot area

    # Row 6: ∫ε̄ₚdz plots
    "epspdz_0": [1.4, fig_height_inches - 12.2 - 1.65*zpanel_size_inches, xpanel_size_inches, ypanel_size_inches],  # Left ∫ε̄ₚdz plot area
    "epspdz_1": [6.3, fig_height_inches - 12.2 - 1.65*zpanel_size_inches, xpanel_size_inches, ypanel_size_inches],  # Right ∫ε̄ₚdz plot area
}

# Convert inches to figure coordinates (0-1)
plot_area_positions = {}
for key, pos_inches in plot_area_positions_inches.items():
    left_inches, bottom_inches, width_inches, height_inches = pos_inches
    plot_area_positions[key] = [
        left_inches / fig_width_inches,    # left in figure coordinates
        bottom_inches / fig_height_inches, # bottom in figure coordinates
        width_inches / fig_width_inches,   # width in figure coordinates
        height_inches / fig_height_inches  # height in figure coordinates
    ]

# Create axes with extra space for labels (will be adjusted after plotting)
# 3D axes for bathymetry
ax_3d = [
    fig.add_axes([0.05, 0.70, 0.42, 0.25], projection="3d"),
    fig.add_axes([0.53, 0.70, 0.42, 0.25], projection="3d")
]

# 2D axes for PV, Ro, ∫εₖdy, ∫ε̄ₖdz, ∫ε̄ₚdy, and ∫ε̄ₚdz
axes = np.array([
    [fig.add_axes([0.05, 0.50, 0.42, 0.18]),
     fig.add_axes([0.53, 0.50, 0.42, 0.18])],
    [fig.add_axes([0.05, 0.30, 0.42, 0.18]),
     fig.add_axes([0.53, 0.30, 0.42, 0.18])],
    [fig.add_axes([0.05, 0.05, 0.42, 0.23]),
     fig.add_axes([0.53, 0.05, 0.42, 0.23])],
    [fig.add_axes([0.05, 0.05, 0.42, 0.18]),
     fig.add_axes([0.53, 0.05, 0.42, 0.18])],
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
yticks = [-500, 0, 500,]

# Row configurations
rows = [
    dict(var="PV", label="Potential vorticity",
         plot_kwargs=dict(vmin=-1.5 * N2_inf * abs(f_0),
                          vmax=+1.5 * N2_inf * abs(f_0),
                          cmap="RdBu_r"),
         xlabel="", remove_xticks=True, remove_xlabel=True,
         plot_type="xy", aspect=None, yticks=yticks, ylabel="y [m]"),
    dict(var="⟨R̄o⟩ᶻ", label="⟨Ro⟩ᶻ",
         plot_kwargs=dict(vmin=-0.4,
                          vmax=+0.4,
                          cmap="RdBu_r"),
         xlabel="x [m]", remove_xticks=False, remove_xlabel=True,
         plot_type="xy", aspect=None, yticks=yticks, ylabel="y [m]"),
    dict(var="∫ε̄ₖdy_normalized", label="∫ε̄ₖdy / W [m²/s³]",
         plot_kwargs=dict(norm=LogNorm(vmin=5e-10, vmax=5e-7),
                          cmap="inferno"),
         xlabel="x [m]", remove_xticks=True, remove_xlabel=True,
         plot_type="xz", aspect=None, yticks=None, ylabel="z [m]", white_colorbar=True),
    dict(var="∫ε̄ₖdz_normalized", label="∫ε̄ₖdz / H [m²/s³]",
         plot_kwargs=dict(norm=LogNorm(vmin=5e-10, vmax=5e-7),
                          cmap="inferno"),
         xlabel="x [m]", remove_xticks=False, remove_xlabel=True,
         plot_type="xy", aspect=None, yticks=yticks, ylabel="y [m]", white_colorbar=True),
    dict(var="∫ε̄ₚdy_normalized", label="∫ε̄ₚdy / W [m²/s³]",
         plot_kwargs=dict(norm=LogNorm(vmin=1e-11, vmax=5e-9),
                          cmap="inferno"),
         xlabel="x [m]", remove_xticks=True, remove_xlabel=True,
         plot_type="xz", aspect=None, yticks=None, ylabel="z [m]", white_colorbar=True),
    dict(var="∫ε̄ₚdz_normalized", label="∫ε̄ₚdz / H [m²/s³]",
         plot_kwargs=dict(norm=LogNorm(vmin=1e-11, vmax=5e-9),
                          cmap="inferno"),
         xlabel="x [m]", remove_xticks=False, remove_xlabel=False,
         plot_type="xy", aspect=None, yticks=yticks, ylabel="y [m]", white_colorbar=True)
]

print("Plotting 2D fields...")
for row_idx, config in enumerate(rows):
    for col_idx, L_val in enumerate(L_values):
        ds = datasets[L_val]

        ax = axes[row_idx, col_idx]

        # Get data
        data = ds[config["var"]]
        im = data.plot(ax=ax, x="x_caa", rasterized=True, add_colorbar=False, **config["plot_kwargs"])

        # Add bathymetry mask for PV and ∫ε̄ₖdz plots
        if row_idx == 0:  # PV plot
            bathy_mask = ds.peripheral_nodes_ccc
            bathy_mask.plot.imshow(ax=ax, cmap="Greys", vmin=0, vmax=1, origin="lower",
                                   alpha=0.25, zorder=2, add_colorbar=False)

        # Set labels and formatting
        if not config.get("remove_xlabel", False):
            ax.set_xlabel(config["xlabel"])
        else:
            ax.set_xlabel("")
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

for row_idx in range(6):
    for col_idx in range(2):
        if row_idx == 0:
            key = f"pv_{col_idx}"
        elif row_idx == 1:
            key = f"ro_{col_idx}"
        elif row_idx == 2:
            key = f"eps_{col_idx}"
        elif row_idx == 3:
            key = f"epsdz_{col_idx}"
        elif row_idx == 4:
            key = f"epspdy_{col_idx}"
        else:  # row_idx == 5
            key = f"epspdz_{col_idx}"
        axes[row_idx, col_idx].set_position(plot_area_positions[key])
#---

#+++ Finalize
delta = H.item() / FWHM.item()
fig.suptitle(f"Ro$_b$ = {datasets['0'].Ro_b.item()}, Fr$_b$ = {datasets['0'].Fr_b.item()}, S$_b$ = {datasets['0'].Slope_Bu.item()}",
             fontsize=14, y=1, x=0.45)
letterize(fig.axes[:14], x=0.05, y=0.72, fontsize=12, bbox=dict(boxstyle="square", facecolor="white", alpha=0.8))

output_path = f"../figures/{simname_base}_dynamics_comparison_L0_vs_L08_dz{resolution}_buffer{buffer}.pdf"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Saved plot to {output_path}")
#---
