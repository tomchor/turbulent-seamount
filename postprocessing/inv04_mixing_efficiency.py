import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
from os.path import basename
import numpy as np
import pynanigans as pn
import xarray as xr
from cycler import cycler
from matplotlib import pyplot as plt
from src.aux00_utils import merge_datasets, condense
from src.aux02_plotting import letterize, create_mc, mscatter

#+++ Define directory and simulation name
path = "../simulations/data/"
simname_base = "seamount"

Rossby_numbers = cycler(Ro_h = [0.2])
Froude_numbers = cycler(Fr_h = [1.25])
L              = cycler(L = [0, 0.05, 0.1, 0.2, 0.4, 0.8])

resolutions    = cycler(dz = [8, 4, 2])
closures       = cycler(closure = [ "DSM",])

paramspace = Rossby_numbers * Froude_numbers * L
configs    = resolutions * closures

runs = paramspace * configs
#---

turb = merge_datasets(runs, base_name=f"turbstats_{simname_base}", verbose=True, add_min_spacings=False,
                      drop_vars=["Δx_min", "Δy_min", "Δz_min", "y_aca",])
turb = turb.reindex(Ro_h = list(reversed(turb.Ro_h)))

#+++ Define new variables
turb["RoFr"] = turb.Ro_h * turb.Fr_h

turb["𝒦ℰ"] = turb["∭⁵⟨Ek′⟩ₜdV"]
turb["𝒫"] = turb["∭⁵SPRdxdy"].sum("j")
turb["ℬ"] = turb["∭⁵⟨w′b′⟩ₜdV"]
turb["𝒜"] = turb["V∞∬⟨Ek′⟩ₜdxdz"]
#---

#+++ Make it legible
turb["𝒦ℰ"].attrs = dict(long_name=r"Int TKE $\mathcal{KE}$")
turb["𝒫"].attrs = dict(long_name=r"Int shear prod rate $\mathcal{P}$")
turb["ℬ"].attrs = dict(long_name=r"Int turbulent buoyancy flux $\mathcal{B}$")
turb["𝒜"].attrs = dict(long_name=r"Int TKE advection out $\mathcal{A}$")
#---

figs = []

turb.plot.scatter(y="ℬ", hue="L", x="dz", xscale="log", yscale="symlog", cmap="bwr")
for ax in plt.gcf().axes[:-1]:
    ax.set_yscale('symlog', linthresh=1e-3)
figs.append(plt.gcf())

#plt.figure()
#turb.plot.scatter(y="𝒦ℰ", hue="L", x="dz", xscale="log", yscale="log", cmap="bwr")
#figs.append(plt.gcf())

plt.figure()
turb.plot.scatter(y="𝒫", hue="L", x="dz", xscale="log", yscale="log", cmap="bwr")
figs.append(plt.gcf())

plt.figure()
turb.plot.scatter(y="𝒜", hue="L", x="dz", xscale="log", yscale="log", cmap="bwr")
figs.append(plt.gcf())

plt.figure()
turb.plot.scatter(y="Δz̃", hue="L", x="dz", xscale="log", yscale="log", cmap="bwr")
figs.append(plt.gcf())

for fig in figs:
    for ax in fig.axes:
        ax.grid(True)
