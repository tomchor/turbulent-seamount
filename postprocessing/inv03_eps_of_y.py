import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
from os.path import basename
import numpy as np
import pynanigans as pn
import xarray as xr
from cycler import cycler
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from src.aux00_utils import merge_datasets, condense, integrate
plt.rcParams["figure.constrained_layout.use"] = True

#+++ Define directory and simulation name
path = "../simulations/data/"
simname_base = "seamount"

Rossby_numbers = cycler(Ro_h = [0.2])
Froude_numbers = cycler(Fr_h = [1.25])
L              = cycler(L = [0, 0.05, 0.1, 0.2, 0.4, 0.8])

resolutions    = cycler(dz = [2])

paramspace = Rossby_numbers * Froude_numbers * L
configs    = resolutions

runs = paramspace * configs
#---

#+++ Read and pre-process datasets
xyza = merge_datasets(runs, base_name=f"xyza.{simname_base}", verbose=True, add_min_spacings=False,
                      open_dataset_kwargs=dict(chunks="auto"))
xyza = xyza.reindex(Ro_h = list(reversed(xyza.Ro_h)))
xyza = xyza.squeeze()
#---

#+++ Condense buffers
distances = [5, 10]
xyza = condense(xyza, ["∫⁵εₖdx", "∫¹⁰εₖdx"], "∫ᵇεₖdx", dimname="buffer", indices=distances)
xyza = condense(xyza, ["∫⁵εₚdx", "∫¹⁰εₚdx"], "∫ᵇεₚdx", dimname="buffer", indices=distances)
#---

#+++ Integrate over z and create new variables
xyza["∬ᵇεₖdxdz"] = integrate(xyza["∫ᵇεₖdx"], dV=xyza.Δz_aac, dims=("z"))
xyza["∬ᵇεₚdxdz"] = integrate(xyza["∫ᵇεₚdx"], dV=xyza.Δz_aac, dims=("z"))

xyza["∬ᵇγdxdz"] = xyza["∬ᵇεₚdxdz"] / (xyza["∬ᵇεₖdxdz"] + xyza["∬ᵇεₚdxdz"])
#---

#+++ Plot
print("Plotting")
xyza["∬ᵇεₖdxdz"].pnplot(x="y", hue="L", row="buffer")
figk = plt.gcf()

xyza["∬ᵇεₚdxdz"].pnplot(x="y", hue="L", row="buffer")
figp = plt.gcf()

xyza["∬ᵇγdxdz"].pnplot(x="y", hue="L", row="buffer")
figg = plt.gcf()
#---

#+++ Prettify
for ax in (figk.axes + figp.axes + figg.axes):
    ax.grid(True)
    ax.axvline(x=0, color="black", linestyle="--")
#---