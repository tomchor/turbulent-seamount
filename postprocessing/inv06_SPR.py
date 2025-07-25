import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
from os.path import basename
import numpy as np
import pynanigans as pn
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

path = "data_post/"
simname_base = "seamount"

flat = xr.open_dataset(path + f"tafields_{simname_base}_α=0.05_Ro_h=0.5_Fr_h=0.2_res=2_closure=AMD_bounded=0.nc")
steep = xr.open_dataset(path + f"tafields_{simname_base}_α=0.2_Ro_h=0.5_Fr_h=0.2_res=2_closure=AMD_bounded=0.nc")

flat  = flat.expand_dims(("α", "f")).assign_coords(α = [0.05],
                                                   x_caa = flat.x_caa / flat.FWMH,
                                                   y_aca = flat.y_aca / flat.FWMH)
steep = steep.expand_dims(("α", "f")).assign_coords(α = [0.2],
                                                    x_caa = steep.x_caa / steep.FWMH,
                                                    y_aca = steep.y_aca / steep.FWMH)


flat["q̄_norm"]  = flat.q̄ / (flat.N2_inf * flat.f_0)
steep["q̄_norm"] = steep.q̄ / (steep.N2_inf * steep.f_0)

intflat = flat.interp(x_caa=steep.x_caa, y_aca=steep.y_aca)
tafields = xr.combine_by_coords([intflat, steep], combine_attrs="drop_conflicts")

tafields["Π"] = tafields.SPR.sum("j")
tafields["Πh"] = tafields.SPR.sel(j=[1, 2]).sum("j")

lim = 9e-7
opts = dict(vmin=-lim, vmax=+lim, cmap="RdBu_r")
tafields.Π.plot(col="α", **opts)
tafields.Πh.plot(col="α", **opts)

tafields.q̄_norm.plot(col="α", robust=True)

tafields["ε̄ₖ"].plot(col="α", norm=LogNorm(), vmin=1e-10, vmax=1e-7)
