import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
from os.path import basename
import numpy as np
import pynanigans as pn
import xarray as xr
from cycler import cycler
from matplotlib import pyplot as plt
from aux00_utils import collect_datasets, form_run_names
from aux02_plotting import letterize, create_mc, mscatter

#+++ Define directory and simulation name
if basename(__file__) != "h00_run_postproc.py":
    path = "simulations/data/"
    simname_base = "tokara"

    resolutions = cycler(res = [4,])
    slopes = cycler(α = [0.05,])
    slopes = cycler(α = [0.2,])
    Rossby_numbers = cycler(Ro_h = [0.08, 0.2, 0.5, 1.25])
    Froude_numbers = cycler(Fr_h = [0.08, 0.2, 0.5, 1.25])
    runs = resolutions * slopes * Rossby_numbers * Froude_numbers
#---

simnames_filtered = list(map(lambda run: form_run_names("tokara", run, sep="_", prefix=""), runs))
bulk = collect_datasets(simnames_filtered, slice_name="bulkstats")
bulk = bulk.reindex(Ro_h = list(reversed(bulk.Ro_h)))
bulk = create_mc(bulk)

#+++ Define new variables
bulk["γ⁵"] = bulk["∭⁵ε̄ₚdV"] / (bulk["∭⁵ε̄ₚdV"] + bulk["∭⁵ε̄ₖdV"])

bulk["H"]  = bulk.α * bulk.L
bulk["ℰₖ"] = bulk["∭⁵ε̄ₖdV"] / (bulk.attrs["V∞"]**3 * bulk.L * bulk.H)
bulk["ℰₚ"] = bulk["∭⁵ε̄ₚdV"] / (bulk.attrs["V∞"]**3 * bulk.L * bulk.H)
#---

#+++ Choose buffers and set some attributes
bulk.Slope_Bu.attrs =  dict(long_name=r"$S_{Bu} = Bu_h^{1/2} = Ro_h / Fr_h$")
bulk["ℰₖ"].attrs = dict(long_name="Normalized integrated\nKE dissipation rate, $\\mathcal{E}_k$")
bulk["ℰₚ"].attrs = dict(long_name="Normalized integrated\nbuoyancy mixing rate, $\\mathcal{E}_p$")
#---

#+++ Create figure
nrows = 2
ncols = 1
size = 3
fig, axes = plt.subplots(ncols=ncols, nrows=nrows,
                         figsize = (2*ncols*size, nrows*size),
                         sharex=True, sharey=False,
                         constrained_layout=True)
axesf = axes.flatten()
#---

#+++ Auxiliary continuous variables
S_Bu = np.logspace(np.log10(bulk["Slope_Bu"].min())+1/3, np.log10(bulk["Slope_Bu"].max()) - 1/3)
rates_curve = 0.1*S_Bu
#---

#+++ Plot stuff
print("Plotting axes 0")
ax = axesf[0]
xvarname = "Slope_Bu"
yvarname = "ℰₖ"
mscatter(x=bulk[xvarname].values.flatten(), y=bulk[yvarname].values.flatten(), color=bulk.color.values.flatten(), markers=bulk.marker.values.flatten(), ax=ax)
ax.set_ylabel(bulk[yvarname].attrs["long_name"]); ax.set_xlabel(bulk[xvarname].attrs["long_name"])
ax.set_xscale("log"); ax.set_yscale("log")
ax.plot(S_Bu, rates_curve, ls="--", label=r"0.1 $S_h$", color="k")

print("Plotting axes 1")
ax = axesf[1]
xvarname = "Slope_Bu"
yvarname = "ℰₚ"
mscatter(x=bulk[xvarname].values.flatten(), y=bulk[yvarname].values.flatten(), color=bulk.color.values.flatten(), markers=bulk.marker.values.flatten(), ax=ax)
ax.set_ylabel(bulk[yvarname].attrs["long_name"]); ax.set_xlabel(bulk[xvarname].attrs["long_name"])
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_ylim(1e-1, 10)
ax.plot(S_Bu, rates_curve, ls="--", label=r"0.1 $S_h$", color="k")
#---

#+++ Prettify and save
for ax in axesf:
    ax.legend(loc="lower right")
    ax.grid(True)
    ax.set_title("")
    ax.set_xlabel("$S_h$")

letterize(axesf, x=0.05, y=0.9, fontsize=14)
fig.savefig(f"figures/dissip_scalings_m{modifier}.pdf")
#---

