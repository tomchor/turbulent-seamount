import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
sys.path.append("/glade/u/home/tomasc/repos/xanimations")
import numpy as np
import pynanigans as pn
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from aux00_utils import open_simulation, adjust_times
from aux02_plotting import BuRd
from cmocean import cm
plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["font.size"] = 9
π = np.pi

path = "simulations/data/"
modifiers = ["-A02-f4", "-A02"]
#modifiers = ["-f8", "-f4",]
alphas = (0.3, 1)
normalized_offsets = (-1/2, 0, 1/2)
V_tokara = 1 # m/s
H_tokara = 500 # meters

fig, axes = plt.subplots(nrows=3, constrained_layout=True, figsize=(10, 8),
                         sharex=True)

for j, modifier in enumerate(modifiers):
    #+++ Open dataset and pick time
    xyz = open_simulation(path+f"xyz.tokara{modifier}.nc",
                          use_inertial_periods = True,
                          topology = "PNN",
                          squeeze = True,
                          load = False,
                          open_dataset_kwargs = dict(chunks=dict(yC="auto", time="auto")),
                          get_grid = False,
                          )
    xyz = adjust_times(xyz, round_times=True)
    xyz = xyz.sel(time=1.5, method="nearest")
    #---

    opts = dict(norm=LogNorm(clip=True), vmin=1e-10, vmax=1e-7, cmap="inferno")

    #+++ Take vertical average
    xyz["ε̄ₖ"]  = (xyz["εₖ"]  * xyz["Δzᶜᶜᶜ"]).pnsum("z") / xyz["Δzᶜᶜᶜ"].pnsum("z")
    xyz["ε̄ₚ"]  = (xyz["εₚ"]  * xyz["Δzᶜᶜᶜ"]).pnsum("z") / xyz["Δzᶜᶜᶜ"].pnsum("z")

    xyz["ℱεₖ"] = xyz["εₖ"].where(xyz.altitude > 4)
    xyz["ℱεₚ"] = xyz["εₚ"].where(xyz.altitude > 4)

    xyz["ℱε̄ₖ"] = (xyz["ℱεₖ"] * xyz["Δzᶜᶜᶜ"]).pnsum("z") / xyz["Δzᶜᶜᶜ"].pnsum("z")
    xyz["ℱε̄ₚ"] = (xyz["ℱεₚ"] * xyz["Δzᶜᶜᶜ"]).pnsum("z") / xyz["Δzᶜᶜᶜ"].pnsum("z")

    xyz["γ"] = xyz["ℱε̄ₚ"] / (xyz["ℱε̄ₚ"] + xyz["ℱε̄ₖ"])
    #---

    #+++ Upscale LES results
    FWMH_tokara = (H_tokara / xyz.H) * xyz.FWMH # m
    ℰₖ_tokara = V_tokara**3 / FWMH_tokara
    ℰₖ_LES = xyz.attrs["V∞"]**3 / (xyz.FWMH)

    xyz["ε̄ₖ_upscaled"]  = xyz["ε̄ₖ"]  * ℰₖ_tokara / ℰₖ_LES
    xyz["ℱε̄ₖ_upscaled"] = xyz["ℱε̄ₖ"] * ℰₖ_tokara / ℰₖ_LES

    xyz["ε̄ₚ_upscaled"]  = xyz["ε̄ₚ"]  * ℰₖ_tokara / ℰₖ_LES
    xyz["ℱε̄ₚ_upscaled"] = xyz["ℱε̄ₚ"] * ℰₖ_tokara / ℰₖ_LES
    #---

    alpha = alphas[j]
    for i, offset in enumerate(normalized_offsets):
        print(f"Plotting {j}-th modifier = {modifier}, {i}-th offset = {offset}")
        color = plt.rcParams["axes.prop_cycle"].by_key()["color"][i]
        xyz_line = xyz.sel(xC=offset*xyz.FWMH, method="nearest")

        label = f"Filtered Vert avg εₖ @ {offset:.1f} FWMH" if j==0 else ""
        xyz_line["ℱε̄ₖ_upscaled"].plot(ax=axes[0], label=label, color=color, alpha=alpha)

        label = f"Filtered Vert avg εₚ @ {offset:.1f} FWMH" if j==0 else ""
        xyz_line["ℱε̄ₚ_upscaled"].plot(ax=axes[1], label=label, color=color, alpha=alpha)

        label = f"Mixing efficiency @ {offset:.1f} FWMH" if j==0 else ""
        xyz_line["γ"].plot(ax=axes[2], label=label, color=color, alpha=alpha)

for i, ax in enumerate(axes):
    if i < 2:
        ax.set_ylim(1e-9, 1e-5)
        ax.set_yscale("log")
    ax.set_title("")
    ax.legend()
    ax.grid(True)

fig.savefig(f"figures/dissipation_comparison.png")
