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


CSM = open_simulation(path+f"iyz.tokara{modifier}.nc",
                      use_inertial_periods = True,
                      topology = "PNN",
                      squeeze = True,
                      load = False,
                      open_dataset_kwargs = dict(chunks="auto"),
                      get_grid = False,
                      )
DSM = open_simulation(path+f"iyz.tokara-DYN{modifier}.nc",
                      use_inertial_periods = True,
                      topology = "PNN",
                      squeeze = True,
                      load = False,
                      open_dataset_kwargs = dict(chunks="auto"),
                      get_grid = False,
                      )
AMD = open_simulation(path+f"iyz.tokara-AMD{modifier}.nc",
                      use_inertial_periods = True,
                      topology = "PNN",
                      squeeze = True,
                      load = False,
                      open_dataset_kwargs = dict(chunks="auto"),
                      get_grid = False,
                      )

CSM = adjust_times(CSM, round_times=True)
DSM = adjust_times(DSM, round_times=True)
AMD = adjust_times(AMD, round_times=True)

CSM = CSM.sel(time=1, method="nearest")
DSM = DSM.sel(time=1, method="nearest")
AMD = AMD.sel(time=1, method="nearest")

opts = dict(norm=LogNorm(clip=True), vmin=1e-10, vmax=1e-7, cmap="inferno", rasterized=True)
fig, axes = plt.subplots(nrows=3, constrained_layout=True, sharey=True, figsize=(14, 7))

for ax, ds in zip(axes, [CSM, DSM, AMD]):
    ds["εₖ"].pnplot(ax=ax, **opts)
