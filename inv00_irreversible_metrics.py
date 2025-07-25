import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
from os.path import basename
import numpy as np
import pynanigans as pn
import xarray as xr
from cycler import cycler
from matplotlib import pyplot as plt
from aux00_utils import merge_datasets, condense
plt.rcParams["figure.constrained_layout.use"] = True

#+++ Define directory and simulation name
path = "simulations/data/"
simname_base = "seamount"

Rossby_numbers = cycler(Ro_h = [0.2])
Froude_numbers = cycler(Fr_h = [1.25])
L              = cycler(L = [0, 20, 40, 80, 160, 320])

resolutions    = cycler(dz = [8, 4, 2])

paramspace = Rossby_numbers * Froude_numbers * L
configs    = resolutions

runs = paramspace * configs
#---

aaaa = merge_datasets(runs, base_name=f"aaaa_{simname_base}", verbose=True, add_min_spacings=False)
aaaa = aaaa.reindex(Ro_h = list(reversed(aaaa.Ro_h)))

fit_filename = f'data_post/bathymetry_powerlaw_fits_{simname_base}.nc'
ds_fit = xr.open_dataset(fit_filename).sel(L=slice(0, 400))
aaaa = xr.merge([aaaa, ds_fit])

#+++ Define new variables
#+++ Condense buffers
distances = [5, 10, 20]
aaaa = condense(aaaa, ["∭⁵ε̄ₚdV", "∭¹⁰ε̄ₚdV", "∭²⁰ε̄ₚdV"], "∭ᵇε̄ₚdV", dimname="buffer", indices=distances)
aaaa = condense(aaaa, ["∭⁵ε̄ₖdV", "∭¹⁰ε̄ₖdV", "∭²⁰ε̄ₖdV"], "∭ᵇε̄ₖdV", dimname="buffer", indices=distances)
#---

aaaa["γ"] = aaaa["∭ᵇε̄ₚdV"] / (aaaa["∭ᵇε̄ₚdV"] + aaaa["∭ᵇε̄ₖdV"])

aaaa["RoFr"] = aaaa.Ro_h * aaaa.Fr_h

hor_scale = 1/aaaa.transition_wavenumber
hor_scale = aaaa.FWHM
aaaa["ℰₖ"] = aaaa["∭ᵇε̄ₖdV"] / (aaaa.attrs["V∞"]**3 * aaaa.FWHM**2 * aaaa.H / hor_scale)
aaaa["ℰₚ"] = aaaa["∭ᵇε̄ₚdV"] / (aaaa.attrs["V∞"]**3 * aaaa.FWHM**2 * aaaa.H / hor_scale)

aaaa["𝒦⁵"] = (aaaa["∭ᵇε̄ₚdV"] / aaaa["N²∞"]) / (aaaa["V∞"] * aaaa.FWHM**2 * aaaa.H**2)
#---

#+++ Make it legible
aaaa["𝒦⁵"].attrs = dict(long_name=r"Norm buoyancy diffusivity $\mathcal{K}$")
#---

aaaa = aaaa.where(aaaa.Slope_Bu==0.16, drop=True).squeeze()

figs = []

# aaaa.plot.scatter(y="ℰₚ", col="buffer", x="dz", hue="L", xscale="log", yscale="log", cmap="bwr")
# figs.append(plt.gcf())

# aaaa.plot.scatter(y="ℰₖ", col="buffer", x="dz", hue="L", xscale="log", yscale="log", cmap="bwr")
# figs.append(plt.gcf())

plt.figure()
aaaa["ℰₖ"].sel(dz=0, method="nearest").plot.scatter(x="L", hue="buffer", cmap="bwr")
figs.append(plt.gcf())

plt.figure()
aaaa["ℰₚ"].sel(dz=0, method="nearest").plot.scatter(x="L", hue="buffer", cmap="bwr")
figs.append(plt.gcf())

for fig in figs:
    for ax in fig.axes[:1]:
        ax.grid(True)
        ax.axvline(x=aaaa.FWHM, color="black", linestyle="--", label="Seamount horz scale FWHM")
    ax.legend(loc="upper right")
