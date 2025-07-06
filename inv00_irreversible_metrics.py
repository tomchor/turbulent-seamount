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

#region Define directory and simulation name
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
#endregion

aaaa = merge_datasets(runs, base_name=f"aaaa_{simname_base}", verbose=True, add_min_spacings=False)
aaaa = aaaa.reindex(Ro_h = list(reversed(aaaa.Ro_h)))

#region Define new variables
#region Condense buffers
distances = [5, 10, 20]
aaaa = condense(aaaa, ["∭⁵ε̄ₚdV", "∭¹⁰ε̄ₚdV", "∭²⁰ε̄ₚdV"], "∭ᵇε̄ₚdV", dimname="buffer", indices=distances)
aaaa = condense(aaaa, ["∭⁵ε̄ₖdV", "∭¹⁰ε̄ₖdV", "∭²⁰ε̄ₖdV"], "∭ᵇε̄ₖdV", dimname="buffer", indices=distances)
#endregion

aaaa["γ"] = aaaa["∭ᵇε̄ₚdV"] / (aaaa["∭ᵇε̄ₚdV"] + aaaa["∭ᵇε̄ₖdV"])

aaaa["RoFr"] = aaaa.Ro_h * aaaa.Fr_h

aaaa["ℰₖ"] = aaaa["∭ᵇε̄ₖdV"] / (aaaa.attrs["V∞"]**3 * aaaa.FWHM * aaaa.H)
aaaa["ℰₚ"] = aaaa["∭ᵇε̄ₚdV"] / (aaaa.attrs["V∞"]**3 * aaaa.FWHM * aaaa.H)

aaaa["𝒦⁵"] = (aaaa["∭ᵇε̄ₚdV"] / aaaa["N²∞"]) / (aaaa["V∞"] * aaaa.FWHM**2 * aaaa.H**2)
#endregion

#region Make it legible
aaaa["𝒦⁵"].attrs = dict(long_name=r"Norm buoyancy diffusivity $\mathcal{K}$")
#endregion

figs = []

aaaa.sel(dz=0, method="nearest").plot.scatter(x="Slope_Bu", y="ℰₖ", hue="L", col="buffer", row="closure", xscale="log", yscale="log", cmap="bwr")
figs.append(plt.gcf())

aaaa.sel(dz=0, method="nearest").plot.scatter(x="Slope_Bu", y="ℰₚ", hue="L", col="buffer", row="closure", xscale="log", yscale="log", cmap="bwr")
figs.append(plt.gcf())

aaaa.sel(buffer=5).plot.scatter(x="Slope_Bu", y="ℰₚ", hue="L", col="dz", row="closure", xscale="log", yscale="log", cmap="bwr")
figs.append(plt.gcf())

aaaa.sel(buffer=5).plot.scatter(x="Slope_Bu", y="ℰₖ", hue="L", col="dz", row="closure", xscale="log", yscale="log", cmap="bwr")
figs.append(plt.gcf())

for fig in figs:
    for ax in fig.axes:
        ax.grid(True)
