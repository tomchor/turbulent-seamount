import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
import pynanigans as pn
from cycler import cycler
from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
plt.rcParams["figure.constrained_layout.use"] = True

postproc_path = "../postprocessing/data/"
resolution = "dz2"

aaad_L00 = xr.open_dataset(postproc_path + f"aaad.seamount_Ro_b0.1_Fr_b1_L0_FWHM500_{resolution}.nc")
aaad_L08 = xr.open_dataset(postproc_path + f"aaad.seamount_Ro_b0.1_Fr_b1_L0.8_FWHM500_{resolution}.nc")

aaad_L00 = aaad_L00.sel(buffer=10).sum("j")
aaad_L08 = aaad_L08.sel(buffer=10).sum("j")

quantities = [
    "∬SPRdydz",
    "∬⟨Ek′⟩ₜdydz",
    # "∬ε̄ₖdydz",
    "∬⟨w′b′⟩ₜdydz",
    # "∬ε̄ₚdydz"
]

nrows = len(quantities)
ncols = 1
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 10))
axes = axes.flatten()

# Plot each quantity
for i, var_name in enumerate(quantities):
    ax = axes[i]

    aaad_L00[var_name].pnplot(x="x", ax=ax)
    aaad_L08[var_name].pnplot(x="x", ax=ax)