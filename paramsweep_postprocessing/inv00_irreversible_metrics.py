import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
from os.path import basename
import numpy as np
import pynanigans as pn
import xarray as xr
from cycler import cycler
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from src.aux00_utils import merge_datasets, condense
plt.rcParams["figure.constrained_layout.use"] = True

#+++ Define directory and simulation name
simname_base = "seamount"

Rossby_numbers = cycler(Ro_h = [0.08, 0.2, 0.5, 1.2])
Froude_numbers = cycler(Fr_h = [0.08, 0.2, 0.5, 1.2])
L              = cycler(L = [0, 0.8])
FWHM           = cycler(FWHM = [500])

resolutions    = cycler(dz = [2])

paramspace = Rossby_numbers * Froude_numbers * L * FWHM
configs    = resolutions

runs = paramspace * configs
#---

aaaa = merge_datasets(runs, base_name=f"aaaa.{simname_base}", verbose=True, add_min_spacings=False)
aaaa = aaaa.reindex(Ro_h = list(reversed(aaaa.Ro_h)))

# fit_filename = f'data/bathymetry_powerlaw_fits_{simname_base}.nc'
# ds_fit = xr.open_dataset(fit_filename).sel(L=slice(0, 400))
# aaaa = xr.merge([aaaa, ds_fit])

#+++ Define new variables
#+++ Condense buffers
distances = [5, 10]
aaaa = condense(aaaa, ["∭⁵ε̄ₚdV", "∭¹⁰ε̄ₚdV"], "∭ᵇε̄ₚdV", dimname="buffer", indices=distances)
aaaa = condense(aaaa, ["∭⁵ε̄ₖdV", "∭¹⁰ε̄ₖdV"], "∭ᵇε̄ₖdV", dimname="buffer", indices=distances)
#---

aaaa["γ"] = aaaa["∭ᵇε̄ₚdV"] / (aaaa["∭ᵇε̄ₚdV"] + aaaa["∭ᵇε̄ₖdV"])

aaaa["RoFr"] = aaaa.Ro_h * aaaa.Fr_h

# hor_scale = 1/aaaa.transition_wavenumber
hor_scale = aaaa.FWHM
aaaa["ℰₖ"] = aaaa["∭ᵇε̄ₖdV"] / (aaaa.attrs["U∞"]**3 * aaaa.FWHM**2 * aaaa.H / hor_scale)
aaaa["ℰₚ"] = aaaa["∭ᵇε̄ₚdV"] / (aaaa.attrs["U∞"]**3 * aaaa.FWHM**2 * aaaa.H / hor_scale)

aaaa["𝒦⁵"] = (aaaa["∭ᵇε̄ₚdV"] / aaaa["N²∞"]) / (aaaa["U∞"] * aaaa.FWHM**2 * aaaa.H**2)
#---

#+++ Make it legible
aaaa["𝒦⁵"].attrs = dict(long_name=r"Norm buoyancy diffusivity $\mathcal{K}$")
#---

# aaaa = aaaa.where(aaaa.Slope_Bu==0.1, drop=True).squeeze()

figs = []

# plt.figure()
aaaa.sel(dz=0, FWHM=500, method="nearest").plot.scatter(y="ℰₖ", x="Slope_Bu", col="buffer", hue="L", cmap="bwr", yscale="log", xscale="log")
figs.append(plt.gcf())

# plt.figure()
aaaa.sel(dz=0, FWHM=500, method="nearest").plot.scatter(y="ℰₚ", x="Slope_Bu", col="buffer", hue="L", cmap="bwr", yscale="log", xscale="log")
figs.append(plt.gcf())

for fig in figs:
    for ax in fig.axes[:]:
        ax.grid(True)
        # ax.axvline(x=1, color="black", linestyle="--", label="Seamount horz scale FWHM")

        # Add 1:1 line
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        min_val = max(min(xlim), min(ylim))
        max_val = min(max(xlim), max(ylim))
        ax.plot([min_val, 1e2*max_val], 1e-2*np.array([min_val, 1e2*max_val]), 'k--', alpha=0.7, label='1:1 line')
    ax.legend()
