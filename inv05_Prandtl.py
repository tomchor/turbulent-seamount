import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
from os.path import basename
import numpy as np
import pynanigans as pn
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

path = "simulations/data/"
simname_base = "seamount"

tti = xr.open_dataset(path + f"tti.{simname_base}_α=0.05_Ro_h=0.5_Fr_h=0.2_res=2_closure=AMD_bounded=0.nc", decode_times=False).squeeze()
tt0 = tti.isel(time=-1)

(tt0.ν / tt0.κ).where(tt0["εₖ"] > 5e-8).plot(vmin=0.4, vmax=0.8)
#(tt0.ν / tt0.κ).plot(vmin=0.4, vmax=0.8)
#tt0.ν.plot.contour(levels=[5e-4, 1e-3], colors="k", linewidths=0.5)
tt0["εₖ"].plot.contour(colors="k", linewidths=0.5, levels=[5e-8, 1e-7, 2e-7])


