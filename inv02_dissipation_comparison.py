import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
sys.path.append("/glade/u/home/tomasc/repos/xanimations")
import numpy as np
import pynanigans as pn
import xarray as xr
from cycler import cycler
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from aux00_utils import open_simulation, adjust_times, aggregate_parameters
from aux02_plotting import BuRd
from cmocean import cm
plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["font.size"] = 9
π = np.pi

#+++ Runs
path = "simulations/data/"
simname_base = "tokara"

resolutions = cycler(dz = [4, 2])
slopes = cycler(α = [0.1,])
Rossby_numbers = cycler(Ro_h = [1.4])
Froude_numbers = cycler(Fr_h = [0.6])
runs = resolutions * slopes * Rossby_numbers * Froude_numbers
#---

alphas = (0.3, 1)
normalized_offsets = (-1/2, 0, 1/2)
V_tokara = 1 # m/s
H_tokara = 500 # meters

fig, axes = plt.subplots(nrows=3, constrained_layout=True, figsize=(10, 8),
                         sharex=True)

for j, modifiers in enumerate(runs):
    simname = f"{simname_base}_" + aggregate_parameters(modifiers, sep="_", prefix="")

    #+++ Open dataset and pick time
    xyzi = open_simulation(path+f"xyzi.{simname}.nc",
                          use_inertial_periods = True,
                          topology = "PNN",
                          squeeze = True,
                          load = False,
                          open_dataset_kwargs = dict(chunks=dict(y_aca="auto", time="auto")),
                          get_grid = False,
                          )
    xyzi = adjust_times(xyzi, round_times=True)
    xyzi = xyzi.sel(time=1.5, method="nearest")
    #---

    opts = dict(norm=LogNorm(clip=True), vmin=1e-10, vmax=1e-7, cmap="inferno")

    #+++ Take vertical average
    xyzi["ε̄ₖ"]  = (xyzi["εₖ"]  * xyzi["Δz_aac"]).pnsum("z") / xyzi["Δz_aac"].pnsum("z")
    xyzi["ε̄ₚ"]  = (xyzi["εₚ"]  * xyzi["Δz_aac"]).pnsum("z") / xyzi["Δz_aac"].pnsum("z")

    xyzi["ℱεₖ"] = xyzi["εₖ"].where(xyzi.altitude > 4)
    xyzi["ℱεₚ"] = xyzi["εₚ"].where(xyzi.altitude > 4)

    xyzi["ℱε̄ₖ"] = (xyzi["ℱεₖ"] * xyzi["Δz_aac"]).pnsum("z") / xyzi["Δz_aac"].pnsum("z")
    xyzi["ℱε̄ₚ"] = (xyzi["ℱεₚ"] * xyzi["Δz_aac"]).pnsum("z") / xyzi["Δz_aac"].pnsum("z")

    xyzi["γ"] = xyzi["ℱε̄ₚ"] / (xyzi["ℱε̄ₚ"] + xyzi["ℱε̄ₖ"])
    #---

    #+++ Upscale LES results
    FWMH_tokara = (H_tokara / xyzi.H) * xyzi.FWMH # m
    ℰₖ_tokara = V_tokara**3 / FWMH_tokara
    ℰₖ_LES = xyzi.attrs["V∞"]**3 / (xyzi.FWMH)

    xyzi["ε̄ₖ_upscaled"]  = xyzi["ε̄ₖ"]  * ℰₖ_tokara / ℰₖ_LES
    xyzi["ℱε̄ₖ_upscaled"] = xyzi["ℱε̄ₖ"] * ℰₖ_tokara / ℰₖ_LES

    xyzi["ε̄ₚ_upscaled"]  = xyzi["ε̄ₚ"]  * ℰₖ_tokara / ℰₖ_LES
    xyzi["ℱε̄ₚ_upscaled"] = xyzi["ℱε̄ₚ"] * ℰₖ_tokara / ℰₖ_LES
    #---

    alpha = alphas[j]
    for i, offset in enumerate(normalized_offsets):
        print(f"Plotting {j}-th run = {simname}, {i}-th offset = {offset}")
        color = plt.rcParams["axes.prop_cycle"].by_key()["color"][i]
        xyzi_line = xyzi.sel(x_caa=offset*xyzi.FWMH, method="nearest")

        label = f"Filtered Vert avg εₖ @ {offset:.1f} FWMH" if j==0 else ""
        xyzi_line["ℱε̄ₖ_upscaled"].plot(ax=axes[0], label=label, color=color, alpha=alpha)

        label = f"Filtered Vert avg εₚ @ {offset:.1f} FWMH" if j==0 else ""
        xyzi_line["ℱε̄ₚ_upscaled"].plot(ax=axes[1], label=label, color=color, alpha=alpha)

        label = f"Mixing efficiency @ {offset:.1f} FWMH" if j==0 else ""
        xyzi_line["γ"].plot(ax=axes[2], label=label, color=color, alpha=alpha)

for i, ax in enumerate(axes):
    if i < 2:
        ax.set_ylim(1e-9, 1e-5)
        ax.set_yscale("log")
    ax.set_title("")
    ax.legend()
    ax.grid(True)
fig.savefig(f"figures/dissipation_comparison.png")
