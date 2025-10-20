import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
from os.path import basename
import numpy as np
import pynanigans as pn
import xarray as xr
from cycler import cycler
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from src.aux00_utils import collect_datasets, merge_datasets
from src.aux02_plotting import letterize, create_mc, mscatter

#+++ Define directory and simulation name
path = "../simulations/data/"
simname_base = "balanus"

Rossby_numbers = cycler(Ro_b = [0.2, 1.25])
Froude_numbers = cycler(Fr_b = [0.2, 1.25])
L              = cycler(L = [0, 300])

resolutions    = cycler(dz = [2,])
closures       = cycler(closure = ["AMD", "CSM", "DSM", "NON"])
closures       = cycler(closure = ["AMD", "CSM",])

paramspace = Rossby_numbers * Froude_numbers * L
configs    = resolutions * closures

runs = paramspace * configs
#---

tafields = merge_datasets(runs, base_name=f"tafields_{simname_base}", verbose=True).squeeze()

tafields = tafields.rename(Δz_min = "Δz")
tafields["Δz"].attrs = dict(units="m")
tafields = tafields.reindex(Ro_b = list(reversed(tafields.Ro_b)))

tafields["ε̄ₖ"].sel(closure="AMD").plot(x="x_caa", y="y_aca", col="Fr_b", row="Ro_b", norm=LogNorm(clip=True), vmin=1e-10, vmax=1e-6)
