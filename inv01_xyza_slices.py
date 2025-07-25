import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
from os.path import basename
import numpy as np
import pynanigans as pn
import xarray as xr
from cycler import cycler
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from src.aux00_utils import merge_datasets
plt.rcParams["figure.constrained_layout.use"] = True

#+++ Define directory and simulation name
path = "simulations/data/"
simname_base = "seamount"

Rossby_numbers = cycler(Ro_h = [0.2])
Froude_numbers = cycler(Fr_h = [1.25])
L              = cycler(L = [0, 20, 40, 80, 160, 320])

resolutions    = cycler(dz = [2])
closures       = cycler(closure = ["DSM"])

paramspace = Rossby_numbers * Froude_numbers * L
configs    = resolutions * closures

runs = paramspace * configs
#---

#+++ Read and pre-process datasets
xyza = merge_datasets(runs, base_name=f"xyza_{simname_base}", verbose=True, add_min_spacings=False)
xyza = xyza.reindex(Ro_h = list(reversed(xyza.Ro_h)))
xyza = xyza.squeeze()

Ĥ = xyza.bottom_height.pnmax(("x", "y")) # Actual height of seamount
q_scale = 5 * xyza.N2_inf * xyza.f_0
xyia = xyza.sel(z_aac=Ĥ/5, method="nearest").sel(L = [0, 40, 320])
#---

#+++ Plot
print("Plotting")
xyia["q̄"].pnplot(x="x", col="L", col_wrap=3, vmin=-q_scale, vmax=q_scale, cmap="RdBu_r")
xyia["ε̄ₖ"].pnplot(x="x", col="L", col_wrap=3, norm=LogNorm(vmin=1e-10, vmax=1e-8))
xyia["ε̄ₚ"].pnplot(x="x", col="L", col_wrap=3, norm=LogNorm(vmin=1e-11, vmax=1e-9))
xyia["R̄o"].pnplot(x="x", col="L", col_wrap=3, vmin=-3, vmax=3, cmap="bwr")
#---