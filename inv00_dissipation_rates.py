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
modifier = "-f8"


regl = open_simulation(path+f"iyz.tokara{modifier}.nc",
                      use_inertial_periods = True,
                      topology = "PNN",
                      squeeze = True,
                      load = False,
                      open_dataset_kwargs = dict(chunks="auto"),
                      get_grid = False,
                      )
larg = open_simulation(path+f"iyz.tokara-large{modifier}.nc",
                      use_inertial_periods = True,
                      topology = "PNN",
                      squeeze = True,
                      load = False,
                      open_dataset_kwargs = dict(chunks="auto"),
                      get_grid = False,
                      )
fast = open_simulation(path+f"iyz.tokara-large-fast{modifier}.nc",
                      use_inertial_periods = True,
                      topology = "PNN",
                      squeeze = True,
                      load = False,
                      open_dataset_kwargs = dict(chunks="auto"),
                      get_grid = False,
                      )
stra = open_simulation(path+f"iyz.tokara-stratified{modifier}.nc",
                      use_inertial_periods = True,
                      topology = "PNN",
                      squeeze = True,
                      load = False,
                      open_dataset_kwargs = dict(chunks="auto"),
                      get_grid = False,
                      )

def nondimensionalize(ds):
    ds = ds.assign_coords(xC=ds.xC / ds.L)
    ds = ds.assign_coords(yC=ds.yC / ds.L)
    ds = ds.assign_coords(zC=ds.zC / ds.H)
    ds["ε̂ₖ"] = ds["εₖ"] / (ds.attrs["V∞"]**3 / ds.L)
    return ds

def regularize(ds):
    ds = adjust_times(ds, round_times=True)
    ds = nondimensionalize(ds)
    ds = ds.sel(time=1, method="nearest")
    return ds

regl = regularize(regl)
larg = regularize(larg)
fast = regularize(fast)
stra = regularize(stra)

opts = dict(norm=LogNorm(clip=True), vmin=1e-5, vmax=1e-1, cmap="inferno", rasterized=True)
fig, axes = plt.subplots(nrows=4, constrained_layout=True, sharey=True, figsize=(14, 7))

for ax, ds in zip(axes, [regl, larg, fast, stra]):
    ds["ε̂ₖ"].pnplot(ax=ax, **opts)
    ax.set_title(f"V∞ = {ds.attrs['V∞']} m/s; L = {ds.L} m; $V_\infty^2/L=$ {ds.attrs["V∞"]**3 / ds.L} m²/s³")
