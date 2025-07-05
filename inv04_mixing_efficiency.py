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

resolutions    = cycler(dz = [2, 4, 8])
closures       = cycler(closure = [ "DSM",])

paramspace = Rossby_numbers * Froude_numbers * L
configs    = resolutions * closures

runs = paramspace * configs
#---

aaaa = merge_datasets(runs, base_name=f"aaaa_{simname_base}", verbose=True, add_min_spacings=False)
turb = merge_datasets(runs, base_name=f"turbstats_{simname_base}", verbose=True, add_min_spacings=False,
                      drop_vars=["Δx_min", "Δy_min", "Δz_min", "y_aca",])

turb = turb.reindex(Ro_h = list(reversed(turb.Ro_h)))

#+++ Define new variables
#+++ Condense buffers
distances = [5, 10, 20]
turb = condense(turb, ["∭⁵ε̄ₚdV", "∭¹⁰ε̄ₚdV", "∭²⁰ε̄ₚdV", ], "∭ᵇε̄ₚdV", dimname="buffer", indices=distances)
turb = condense(turb, ["∭⁵ε̄ₖdV", "∭¹⁰ε̄ₖdV", "∭²⁰ε̄ₖdV", ], "∭ᵇε̄ₖdV", dimname="buffer", indices=distances)
#---

turb["γ"] = turb["∭ᵇε̄ₚdV"] / (turb["∭ᵇε̄ₚdV"] + turb["∭ᵇε̄ₖdV"])

turb["RoFr"] = turb.Ro_h * turb.Fr_h

turb["𝒦ℰ"] = turb["∭⁵⟨Ek′⟩ₜdV"]
turb["𝒫"] = turb["∬⁵Πdxdy"]
turb["ℬ"] = turb["∭⁵⟨w′b′⟩ₜdV"]

turb["ℰₖ"] = turb["∭ᵇε̄ₖdV"] / (turb.attrs["V∞"]**3 * turb.FWHM * turb.H)
turb["ℰₚ"] = turb["∭ᵇε̄ₚdV"] / (turb.attrs["V∞"]**3 * turb.FWHM * turb.H)

turb["𝒦⁵"] = (turb["∭ᵇε̄ₚdV"] / turb["N²∞"]) / (turb["V∞"] * turb.FWHM**2 * turb.H**2)
#---

#+++ Make it legible
turb["𝒦ℰ"].attrs = dict(long_name=r"Norm TKE $\mathcal{KE}$")
turb["𝒦⁵"].attrs = dict(long_name=r"Norm buoyancy diffusivity $\mathcal{K}$")
turb["𝒫"].attrs = dict(long_name=r"Norm shear prod rate $\mathcal{P}$")
#---

figs = []

#turb.sel(dz=0, method="nearest").plot.scatter(x="Slope_Bu", y="γ", hue="L", col="buffer", row="closure", xscale="log", yscale="log", cmap="bwr")
#figs.append(plt.gcf())

#turb.plot.scatter(x="Slope_Bu", y="ℬ", hue="L", col="dz", row="closure", xscale="log", yscale="symlog", cmap="bwr")
#for ax in plt.gcf().axes[:-1]:
#    ax.set_yscale('symlog', linthresh=1e-3)
#figs.append(plt.gcf())

#turb.plot.scatter(x="RoFr", y="𝒦⁵", hue="L", col="dz", row="closure", xscale="log", yscale="log", cmap="bwr")
#figs.append(plt.gcf())

turb.plot.scatter(x="Slope_Bu", y="𝒦ℰ", hue="L", col="dz", row="closure", xscale="log", yscale="log", cmap="bwr")
figs.append(plt.gcf())

#turb.plot.scatter(x="Slope_Bu", y="𝒫", hue="L", col="dz", row="closure", xscale="log", yscale="log", cmap="bwr")
#figs.append(plt.gcf())

#turb.sel(dz=0, method="nearest").plot.scatter(x="Slope_Bu", y="ℰₖ", hue="L", col="buffer", row="closure", xscale="log", yscale="log", cmap="bwr")
#figs.append(plt.gcf())

#turb.sel(dz=0, method="nearest").plot.scatter(x="Slope_Bu", y="ℰₚ", hue="L", col="buffer", row="closure", xscale="log", yscale="log", cmap="bwr")
#figs.append(plt.gcf())

#turb.sel(buffer=5).plot.scatter(x="Slope_Bu", y="ℰₚ", hue="L", col="dz", row="closure", xscale="log", yscale="log", cmap="bwr")
#figs.append(plt.gcf())

for fig in figs:
    for ax in fig.axes:
        ax.grid(True)
