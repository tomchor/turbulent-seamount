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
path = "simulations/data/"
simname_base = "tokara"

slopes         = cycler(α = [0.05, 0.2])
Rossby_numbers = cycler(Ro_h = [1.4])
Froude_numbers = cycler(Fr_h = [0.6])

resolutions = cycler(res = [8, 4, 2,])
closures       = cycler(closure = ["AMD", "CSM", "DSM", "NON"])
bcs            = cycler(bounded = [0])

paramspace = slopes * Rossby_numbers * Froude_numbers
configs    = resolutions * closures * bcs

runs = paramspace * configs
#---

simnames_filtered = list(map(lambda run: form_run_names("tokara", run, sep="_", prefix=""), runs))

dslist = []
for sim_number, simname in enumerate(simnames_filtered):
    #+++ Open volume-integrated output
    fname = f"bulkstats_{simname}.nc"
    print(f"\nOpening {fname}")
    ds = xr.open_dataset(f"data_post/{fname}", chunks=dict(time="auto", L="auto"))
    #---

    #+++ Calculate resolutions before they get thrown out
    if "Δx_min" not in ds.keys(): ds["Δx_min"] = ds["Δxᶜᶜᶜ"].where(ds["Δxᶜᶜᶜ"] > 0).min().values
    if "Δy_min" not in ds.keys(): ds["Δy_min"] = ds["Δyᶜᶜᶜ"].where(ds["Δyᶜᶜᶜ"] > 0).min().values
    if "Δz_min" not in ds.keys(): ds["Δz_min"] = ds["Δzᶜᶜᶜ"].where(ds["Δzᶜᶜᶜ"] > 0).min().values
    #---

    #+++ Create auxiliary variables and organize them into a Dataset
    if "PV" in ds.variables.keys():
        ds["PV_norm"] = ds.PV / (ds.N2_inf * ds.f_0)
    ds["simulation"] = simname
    ds["sim_number"] = sim_number
    ds["f₀"] = ds.f_0
    ds["N²∞"] = ds.N2_inf
    ds = ds.expand_dims(("α", "Δz", "closure")).assign_coords(α=[ds.α],
                                                              Δz=[np.round(ds.Δz_min, decimals=4)],
                                                              closure=[ds.closure])
    dslist.append(ds)
    #---

bulk = xr.combine_by_coords(dslist, combine_attrs="drop_conflicts")
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
