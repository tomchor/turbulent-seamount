import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
from os.path import basename
import numpy as np
import pynanigans as pn
import xarray as xr
from cycler import cycler
from matplotlib import pyplot as plt
from aux00_utils import collect_datasets, merge_datasets
from aux02_plotting import letterize, create_mc, mscatter

#+++ Define directory and simulation name
path = "simulations/data/"
simname_base = "seamount"

slopes         = cycler(α = [0.05, 0.2])
Rossby_numbers = cycler(Ro_h = [0.2, 1.25])
Froude_numbers = cycler(Fr_h = [0.2, 1.25])

resolutions    = cycler(dz = [8, 4, 2,])
closures       = cycler(closure = ["AMD", "CSM", "DSM", "NON"])
closures       = cycler(closure = ["AMD", "DSM"])

paramspace = slopes * Rossby_numbers * Froude_numbers
configs    = resolutions * closures

runs = paramspace * configs
#---

bulk = merge_datasets(runs, base_name=f"bulkstats_{simname_base}", verbose=True)
bulk = bulk.rename(Δz_min = "Δz")
bulk["Δz"].attrs = dict(units="m")

#+++ Define new variables
bulk["γ⁵"] = bulk["∭⁵ε̄ₚdV"] / (bulk["∭⁵ε̄ₚdV"] + bulk["∭⁵ε̄ₖdV"])

bulk["H"]  = bulk.α * bulk.L

bulk["𝒦"] = bulk["⟨∬⁵Ek′dxdy⟩ₜ"]
bulk["𝒫"] = bulk["⟨∬⁵Πdxdy⟩ₜ"]

bulk["ℰₖ"] = bulk["∭⁵ε̄ₖdV"] / (bulk.attrs["V∞"]**3 * bulk.L * bulk.H)
bulk["ℰₚ"] = bulk["∭⁵ε̄ₚdV"] / (bulk.attrs["V∞"]**3 * bulk.L * bulk.H)
#---

#+++ Make it legible
bulk["𝒦"].attrs = dict(long_name=r"Norm TKE $\mathcal{K}$")
bulk["𝒫"].attrs = dict(long_name=r"Norm shear prod rate $\mathcal{P}$")
#---

figs = []

bulk["𝒦"].plot(col="α", x="Δz", hue="closure", marker="o", linestyle="", sharey=False)
figs.append(plt.gcf())

bulk["𝒫"].plot(col="α", x="Δz", hue="closure", marker="o", linestyle="", sharey=False)
figs.append(plt.gcf())

bulk["ℰₖ"].plot(col="α", x="Δz", hue="closure", marker="o", linestyle="", yscale="log", ylim=(5e-2, 3))
figs.append(plt.gcf())

bulk["ℰₚ"].plot(col="α", x="Δz", hue="closure", marker="o", linestyle="", yscale="log", ylim=(5e-2, 3))
figs.append(plt.gcf())

bulk["γ⁵"].plot(col="α", x="Δz", hue="closure", marker="o", linestyle="", ylim=(0, None))
figs.append(plt.gcf())
for fig in figs:
    for ax in fig.axes:
        ax.grid(True)
