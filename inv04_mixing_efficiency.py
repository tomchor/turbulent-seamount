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
closures       = cycler(closure = ["AMD", "CSM", "DSM"])

paramspace = slopes * Rossby_numbers * Froude_numbers
configs    = resolutions * closures

runs = paramspace * configs
#---

bulk = merge_datasets(runs, base_name=f"bulkstats_{simname_base}", verbose=True)
bulk = bulk.rename(Δz_min = "Δz")
bulk["Δz"].attrs = dict(units="m")
bulk = bulk.reindex(Ro_h = list(reversed(bulk.Ro_h)))

#+++ Define new variables
bulk["γ⁵"] = bulk["∭⁵ε̄ₚdV"] / (bulk["∭⁵ε̄ₚdV"] + bulk["∭⁵ε̄ₖdV"])
bulk["γ¹⁰"] = bulk["∭¹⁰ε̄ₚdV"] / (bulk["∭¹⁰ε̄ₚdV"] + bulk["∭¹⁰ε̄ₖdV"])

bulk["H"]  = bulk.α * bulk.L
bulk["RoFr"] = bulk.Ro_h * bulk.Fr_h

bulk["𝒦ℰ"] = bulk["⟨∬⁵Ek′dxdy⟩ₜ"]
bulk["𝒫"] = bulk["⟨∬⁵Πdxdy⟩ₜ"]

bulk["ℰₖ"] = bulk["∭⁵ε̄ₖdV"] / (bulk.attrs["V∞"]**3 * bulk.L * bulk.H)
bulk["ℰₚ"] = bulk["∭⁵ε̄ₚdV"] / (bulk.attrs["V∞"]**3 * bulk.L * bulk.H)

bulk["𝒦⁵"] = (bulk["∭⁵ε̄ₚdV"] / bulk["N²∞"]) / (bulk["V∞"] * bulk.L**2 * bulk.H**2)
#---

#+++ Make it legible
bulk["𝒦ℰ"].attrs = dict(long_name=r"Norm TKE $\mathcal{KE}$")
bulk["𝒦⁵"].attrs = dict(long_name=r"Norm buoyancy diffusivity $\mathcal{K}$")
#bulk["𝒦¹⁰"].attrs = dict(long_name=r"Norm buoyancy diffusivity $\mathcal{K}$")
bulk["𝒫"].attrs = dict(long_name=r"Norm shear prod rate $\mathcal{P}$")
#---

figs = []

bulk.plot.scatter(x="Slope_Bu", y="γ⁵", hue="α", col="dz", row="closure", xscale="log", yscale="log", cmap="bwr")
figs.append(plt.gcf())

bulk.plot.scatter(x="RoFr", y="𝒦⁵", hue="α", col="dz", row="closure", xscale="log", yscale="log", cmap="bwr")
figs.append(plt.gcf())

bulk.plot.scatter(x="Slope_Bu", y="𝒫", hue="α", col="dz", row="closure", xscale="log", yscale="log", cmap="bwr")
figs.append(plt.gcf())

bulk.plot.scatter(x="Slope_Bu", y="ℰₖ", hue="α", col="dz", row="closure", xscale="log", yscale="log", cmap="bwr")
figs.append(plt.gcf())

for fig in figs:
    for ax in fig.axes:
        ax.grid(True)
