import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
import numpy as np
import pynanigans as pn
import xarray as xr
from cycler import cycler
from matplotlib import pyplot as plt
from aux00_utils import open_simulation
from cmocean import cm

path = "simulations/data/"
#iyz = open_simulation(path + "iyz.tokara_res=2_α=0.2_Ro_h=1.4_Fr_h=0.6.nc",
#                      use_inertial_periods = True,
#                      topology = "PNN",
#                      squeeze = True,
#                      load = False,
#                      open_dataset_kwargs = dict(chunks=dict(y_aca="auto", time="auto")),
#                      get_grid = False,
#                      )
#iyz.v.sel(time=[3, 3.5, 4], method="nearest").pnplot(y="z", row="time", robust=True)
#iyz["∂u∂z"].sel(time=[3, 3.5, 4], method="nearest").pnplot(y="z", row="time", robust=True)

tokara = open_simulation(path + "xyz.tokara_res=2_α=0.2_Ro_h=1.4_Fr_h=0.6.nc",
                         use_inertial_periods = True,
                         topology = "PNN",
                         squeeze = True,
                         load = False,
                         open_dataset_kwargs = dict(chunks=dict(y_aca="auto", time="auto")),
                         get_grid = False,
                         )
depths = np.linspace(0.1, 0.9, 5) * tokara.H 

coupled = open_simulation(path + "xyz.tokara_res=2_α=0.2_Ro_h=0.08_Fr_h=1.25_closure=AMD_bounded=0.nc",
                          use_inertial_periods = True,
                          topology = "PNN",
                          squeeze = True,
                          load = False,
                          open_dataset_kwargs = dict(chunks=dict(y_aca="auto", time="auto")),
                          get_grid = False,
                          )

tokara.PV.sel(z_aac=depths, time=np.inf, method="nearest").pnplot(y="x", row="z", robust=True)
coupled.PV.sel(z_aac=depths, time=np.inf, method="nearest").pnplot(y="x", row="z", robust=True)

