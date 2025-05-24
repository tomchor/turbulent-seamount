import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
from os.path import basename
import numpy as np
import pynanigans as pn
import xarray as xr
from cycler import cycler
from matplotlib import pyplot as plt
from aux00_utils import merge_datasets, condense
from aux02_plotting import letterize, create_mc, mscatter

#+++ Define directory and simulation name
path = "simulations/data/"
simname_base = "seamount"

Rossby_numbers = cycler(Ro_h = [0.2, 1.25])
Froude_numbers = cycler(Fr_h = [0.2, 1.25])
L              = cycler(L = [0, 300])

resolutions    = cycler(dz = [4, 2,])
closures       = cycler(closure = ["AMD", "CSM",])

paramspace = Rossby_numbers * Froude_numbers * L
configs    = resolutions * closures

runs = paramspace * configs
#---

bulk = merge_datasets(runs, base_name=f"bulkstats_{simname_base}", verbose=True)
bulk = bulk.rename(Δz_min = "Δz")
bulk["Δz"].attrs = dict(units="m")
bulk = bulk.reindex(Ro_h = list(reversed(bulk.Ro_h)))

#+++ Define new variables
#+++ Condense buffers
distances = [5, 10, 20]
bulk = condense(bulk, ["∭⁵ε̄ₚdV", "∭¹⁰ε̄ₚdV", "∭²⁰ε̄ₚdV", ], "∭ᵇε̄ₚdV", dimname="buffer", indices=distances)
bulk = condense(bulk, ["∭⁵ε̄ₖdV", "∭¹⁰ε̄ₖdV", "∭²⁰ε̄ₖdV", ], "∭ᵇε̄ₖdV", dimname="buffer", indices=distances)
#---

bulk["γ"] = bulk["∭ᵇε̄ₚdV"] / (bulk["∭ᵇε̄ₚdV"] + bulk["∭ᵇε̄ₖdV"])

bulk["RoFr"] = bulk.Ro_h * bulk.Fr_h

bulk["𝒦ℰ"] = bulk["⟨∬⁵Ek′dxdy⟩ₜ"]
bulk["𝒫"] = bulk["⟨∬⁵Πdxdy⟩ₜ"]
bulk["ℬ"] = bulk["⟨∬⁵w′b′dxdy⟩ₜ"]

bulk["ℰₖ"] = bulk["∭ᵇε̄ₖdV"] / (bulk.attrs["V∞"]**3 * bulk.FWHM * bulk.H)
bulk["ℰₚ"] = bulk["∭ᵇε̄ₚdV"] / (bulk.attrs["V∞"]**3 * bulk.FWHM * bulk.H)

bulk["𝒦⁵"] = (bulk["∭ᵇε̄ₚdV"] / bulk["N²∞"]) / (bulk["V∞"] * bulk.FWHM**2 * bulk.H**2)
#---

#+++ Make it legible
bulk["𝒦ℰ"].attrs = dict(long_name=r"Norm TKE $\mathcal{KE}$")
bulk["𝒦⁵"].attrs = dict(long_name=r"Norm buoyancy diffusivity $\mathcal{K}$")
bulk["𝒫"].attrs = dict(long_name=r"Norm shear prod rate $\mathcal{P}$")
#---

figs = []

bulk.sel(dz=0, method="nearest").plot.scatter(x="Slope_Bu", y="γ", hue="L", col="buffer", row="closure", xscale="log", yscale="log", cmap="bwr")
figs.append(plt.gcf())

#bulk.plot.scatter(x="Slope_Bu", y="ℬ", hue="L", col="dz", row="closure", xscale="log", yscale="symlog", cmap="bwr")
#for ax in plt.gcf().axes[:-1]:
#    ax.set_yscale('symlog', linthresh=1e-3)
#figs.append(plt.gcf())

#bulk.plot.scatter(x="RoFr", y="𝒦⁵", hue="L", col="dz", row="closure", xscale="log", yscale="log", cmap="bwr")
#figs.append(plt.gcf())

#bulk.plot.scatter(x="Slope_Bu", y="𝒦ℰ", hue="L", col="dz", row="closure", xscale="log", yscale="log", cmap="bwr")
#figs.append(plt.gcf())

#bulk.plot.scatter(x="Slope_Bu", y="𝒫", hue="L", col="dz", row="closure", xscale="log", yscale="log", cmap="bwr")
#figs.append(plt.gcf())

bulk.sel(dz=0, method="nearest").plot.scatter(x="Slope_Bu", y="ℰₚ", hue="L", col="buffer", row="closure", xscale="log", yscale="log", cmap="bwr")
figs.append(plt.gcf())

for fig in figs:
    for ax in fig.axes:
        ax.grid(True)
