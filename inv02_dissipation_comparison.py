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
modifier = "-f2"


iyz = open_simulation(path+f"iyz.tokara{modifier}.nc",
                      use_inertial_periods = True,
                      topology = "PNN",
                      squeeze = True,
                      load = False,
                      open_dataset_kwargs = dict(chunks="auto"),
                      get_grid = False,
                      )

xyz = open_simulation(path+f"xyz.tokara{modifier}.nc",
                      use_inertial_periods = True,
                      topology = "PNN",
                      squeeze = True,
                      load = False,
                      open_dataset_kwargs = dict(chunks="auto"),
                      get_grid = False,
                      )

iyz = adjust_times(iyz, round_times=True)
iyz = iyz.sel(time=3, method="nearest")

iyz["altitude"] = xyz.altitude.interp(xC=iyz.xC, yC=iyz.yC, zC=iyz.zC)

iyz["ℱεₖ"] = iyz["εₖ"].where(iyz.altitude > 4)

opts = dict(norm=LogNorm(clip=True), vmin=1e-10, vmax=1e-7, cmap="inferno")

iyz["ε̄ₖ"]  = (iyz["εₖ"]  * iyz["Δzᶜᶜᶜ"]).pnsum("z") / iyz["Δzᶜᶜᶜ"].pnsum("z")
iyz["ℱε̄ₖ"] = (iyz["ℱεₖ"] * iyz["Δzᶜᶜᶜ"]).pnsum("z") / iyz["Δzᶜᶜᶜ"].pnsum("z")

V = 1 # m/s
FWMH = 10_000 # m
ℰₖ_tokara = V**3 / FWMH
ℰₖ_LES = iyz.attrs["V∞"]**3 / (iyz.L * (2*np.log(2)))

fig, ax = plt.subplots(ncols=1, constrained_layout=True, sharey=True, figsize=(10, 4))

(iyz["ε̄ₖ"] * ℰₖ_tokara / ℰₖ_LES).plot(ax=ax, ylim=(1e-9, 1e-5), yscale="log", label="Vert avg εₖ")
(iyz["ℱε̄ₖ"] * ℰₖ_tokara / ℰₖ_LES).plot(ax=ax, label="Filtered Vert avg εₖ")
ax.legend()

