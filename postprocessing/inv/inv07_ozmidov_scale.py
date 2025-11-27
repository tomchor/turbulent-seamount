import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
from os.path import basename
import numpy as np
import pynanigans as pn
import xarray as xr
from cycler import cycler
from matplotlib import pyplot as plt
plt.rcParams['figure.constrained_layout.use'] = True

import matplotlib as mpl
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["r", "b"])

from src.aux00_utils import collect_datasets, merge_datasets
from src.aux02_plotting import letterize, create_mc, mscatter

#+++ Define directory and simulation name
path = "../simulations/data/"
simname_base = "balanus"

slopes         = cycler(α = [0.05, 0.2])
Rossby_numbers = cycler(Ro_b = [0.2, 1.25])
Froude_numbers = cycler(Fr_b = [0.2, 1.25])

resolutions    = cycler(dz = [8, 4, 2,])
closures       = cycler(closure = ["AMD", "CSM", "DSM", "NON"])
closures       = cycler(closure = ["AMD", "CSM", "DSM"])

paramspace = slopes * Rossby_numbers * Froude_numbers
configs    = resolutions * closures

runs = paramspace * configs
#---

bulk = merge_datasets(runs, base_name=f"turbstats_{simname_base}", verbose=True)
bulk = bulk.rename(Δz_min = "Δz")
bulk["Δz"].attrs = dict(units="m")

for closure in bulk.closure:
    bulk.sel(closure=closure).plot.scatter(col="Fr_b", row="Ro_b", x="Δz", y="Δz̃", hue="α", cmap=mpl.colors.ListedColormap(["red", "blue"]))
    fig = plt.gcf()
    fig.suptitle(closure.item())
    for ax in fig.axes:
        ax.axhline(y=1, ls="--", c="k")
        ax.grid(True)
