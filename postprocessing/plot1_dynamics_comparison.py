import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
from matplotlib.gridspec import GridSpec
from src.aux00_utils import open_simulation
from src.aux02_plotting import letterize

plt.rcParams["figure.constrained_layout.use"] = True

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

# Load data for both L values
L_values = ["0", "0.8"]
datasets = {}

for L_val in L_values:
    simname = f"{simname_base}_Ro_b{Ro_b}_Fr_b{Fr_b}_L{L_val}_dz{resolution}"

    xyzi = open_simulation(f"{simdata_path}xyzi.{simname}.nc", **dataset_opts)
    aaad = open_simulation(f"{postproc_path}aaad.{simname}.nc", **avgd_opts).sel(buffer=buffer)

    ds = xr.merge([xyzi, aaad]).sel(time=np.inf, method="nearest")
    datasets[L_val] = ds

# Extract parameters
H = datasets["0"].H
FWHM = datasets["0"].FWHM
extent = 1.3 * FWHM
#---

#+++ Create figure
fig = plt.figure(figsize=(14, 10))
gs = GridSpec(3, 2, figure=fig, height_ratios=[1.4, 0.7, 0.7], hspace=0)

# 3D axes for bathymetry
ax_3d = [fig.add_subplot(gs[0, i], projection="3d") for i in range(2)]

# 2D axes for PV and Ro
axes = np.array([[fig.add_subplot(gs[1, i]) for i in range(2)],
                 [fig.add_subplot(gs[2, i]) for i in range(2)]])
#---

#+++ Plot 3D bathymetry
ls = LightSource(azdeg=270, altdeg=45)

for i, L_val in enumerate(L_values):
    ds = datasets[L_val]
    bathy = ds["bottom_height"].pnsel(x=slice(-extent, +extent), y=slice(-extent, +extent))

    x, y = np.meshgrid(bathy.x.values, bathy.y.values)
    rgb = ls.shade(bathy.values, cmap=plt.cm.gist_earth, vert_exag=0.1, blend_mode="soft")

    ax_3d[i].plot_surface(x, y, bathy.values, rstride=2, cstride=2, facecolors=rgb,
                          linewidth=0, antialiased=False, shade=False, rasterized=True)
    ax_3d[i].set_xlabel("x [m]")
    ax_3d[i].set_ylabel("y [m]")
    ax_3d[i].set_zlabel("z [m]")
    ax_3d[i].set_title(f"L/W = {ds.L.item()}")
    ax_3d[i].view_init(elev=25, azim=135)
    ax_3d[i].set_box_aspect((1, 1, 0.3))
#---

#+++ Plot 2D fields
yticks = [-1000, -500, 0, 500, 1000]

# Row configurations
rows = [
    dict(var="PV", label="Potential vorticity", cmap="RdBu_r",
         get_data=lambda ds: ds["PV"].pnsel(z=H/3, method="nearest"),
         vmin=lambda ds: -1.5 * ds.N2_inf * abs(ds.f_0),
         vmax=lambda ds: 1.5 * ds.N2_inf * abs(ds.f_0),
         xlabel="", remove_xticks=True),
    dict(var="⟨R̄o⟩ᶻ", label="⟨Ro⟩ᶻ", cmap="RdBu_r",
         get_data=lambda ds: ds["⟨R̄o⟩ᶻ"],
         vmin=lambda ds: -0.4,
         vmax=lambda ds: 0.4,
         xlabel="x [m]", remove_xticks=False)
]

print("Plotting 2D fields...")
for row_idx, config in enumerate(rows):
    for col_idx, L_val in enumerate(L_values):
        ds = datasets[L_val]
        ax = axes[row_idx, col_idx]

        data = config["get_data"](ds)
        vmin = config["vmin"](ds)
        vmax = config["vmax"](ds)

        im = data.plot.imshow(ax=ax, x="x_caa", cmap=config["cmap"],
                              vmin=vmin, vmax=vmax, add_colorbar=False, rasterized=True)

        # Add bathymetry mask for PV plot
        if row_idx == 0:
            bathy_mask = ds.peripheral_nodes_ccc.pnsel(z=H/3, method="nearest")
            bathy_mask.plot.imshow(ax=ax, cmap="Greys", vmin=0, vmax=1, origin="lower",
                                   alpha=0.25, zorder=2, add_colorbar=False)

        ax.set_xlabel(config["xlabel"])
        ax.set_ylabel("y [m]" if col_idx == 0 else "")
        ax.set_yticks(yticks)
        ax.set_aspect("equal")
        ax.set_title("")

        if config["remove_xticks"]:
            ax.set_xticklabels([])
        if col_idx > 0:
            ax.set_yticklabels([])

    # Add colorbar
    cax = axes[row_idx, 1].inset_axes([0.75, 0.1, 0.03, 0.8],
                                      transform=axes[row_idx, 1].transAxes, clip_on=False)
    plt.colorbar(im, cax=cax, orientation="vertical", label=config["label"])
#---

#+++ Finalize
delta = H.item() / FWHM.item()
fig.suptitle(f"Ro$_b$ = {datasets['0'].Ro_b.item()}, Fr$_b$ = {datasets['0'].Fr_b.item()}, $\delta$ = {delta:.1f}",
             fontsize=14, y=0.995)
letterize(fig.axes[:6], x=0.05, y=0.9, fontsize=12,
          bbox=dict(boxstyle="square", facecolor="white", alpha=0.8))

output_path = f"../figures/{simname_base}_dynamics_comparison_L0_vs_L08_dz{resolution}_buffer{buffer}.pdf"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Saved plot to {output_path}")
#---
