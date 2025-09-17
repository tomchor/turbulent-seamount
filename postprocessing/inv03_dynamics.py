import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import pynanigans as pn
from src.aux00_utils import open_simulation

plt.rcParams["figure.constrained_layout.use"] = True

#+++ Load datasets
print("Loading xyzi datasets...")
path = "../simulations/data/"

ds_L00 = open_simulation(path + "xyzi.seamount_Ro_h0.1_Fr_h1_L0_FWHM500_dz2.nc",
                           use_advective_periods=True,
                           squeeze=True,
                           load=False,
                           get_grid=False,
                           open_dataset_kwargs=dict(chunks="auto"))

ds_L08 = open_simulation(path + "xyzi.seamount_Ro_h0.1_Fr_h1_L0.8_FWHM500_dz2.nc",
                           use_advective_periods=True,
                           squeeze=True,
                           load=False,
                           get_grid=False,
                           open_dataset_kwargs=dict(chunks="auto"))
#---

#+++ Create new variables
for ds in [ds_L00, ds_L08]:
    ds["dUdz"] = np.sqrt(ds["∂u∂z"]**2 + ds["∂v∂z"]**2)
#---

#+++ Create 4x2 subplot grid
print("Creating subplot grid")
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(18, 18), sharex=True)
#---

#+++ Plot PV for both cases
print("Plotting PV")
sel_opts = dict(z=ds_L00.H / 3, time=20, method="nearest")
datasets = [(ds_L00, "0"), (ds_L08, "0.8")]

PV_inf = ds_L00.N2_inf * ds_L00.f_0
for i, (ds, L_str) in enumerate(datasets):
    ax = axes[0, i]
    ds.PV.pnsel(**sel_opts).pnplot(ax=ax, x="x", y="y",
                                   cmap="RdBu_r", robust=True,
                                   add_colorbar=True,
                                   rasterized=True,
                                   vmin = -1.5*PV_inf,
                                   vmax = +1.5*PV_inf)
    ax.set_title(f"PV, L/FWHM = {L_str}")
#---

#+++ Plot Ro for both cases
print("Plotting Ro")
for i, (ds, L_str) in enumerate(datasets):
    ax = axes[1, i]
    ds.Ro.pnsel(**sel_opts).pnplot(ax=ax, x="x", y="y",
                                   cmap="RdBu_r", robust=True,
                                   add_colorbar=True,
                                   rasterized=True,
                                   vmin = -3,
                                   vmax = +3)
    ax.set_title(f"Rossby number, L/FWHM = {L_str}")
#---

#+++ Plot u velocity xz cross-sections
print("Plotting u velocity xz cross-sections")
# Select middle y slice for xz cross-section
xz_sel_opts = dict(y=-300, time=20, method="nearest")

for i, (ds, L_str) in enumerate(datasets):
    ax = axes[2, i]
    ds.u.pnsel(**xz_sel_opts).pnplot(ax=ax, x="x", y="z",
                                     cmap="RdBu_r", robust=True,
                                     add_colorbar=True,
                                     rasterized=True,
                                     vmin = -1.5*ds.attrs["U∞"],
                                     vmax = +1.5*ds.attrs["U∞"])
    ax.set_title(f"u velocity (xz), L/FWHM = {L_str}")
#---

#+++ Plot dUdz xz cross-sections
print("Plotting dUdz xz cross-sections")
for i, (ds, L_str) in enumerate(datasets):
    ax = axes[3, i]
    ds.dUdz.pnsel(**xz_sel_opts).pnplot(ax=ax, x="x", y="z",
                                        cmap="plasma", robust=True,
                                        add_colorbar=True,
                                        rasterized=True,
                                        vmin = 0,
                                        vmax = 5e-3)
    ax.set_title(f"Vertical shear |dU/dz| (xz), L/FWHM = {L_str}")
#---

#+++ Save
fig.savefig("../figures/dynamics_comparison.png", dpi=300, bbox_inches="tight")
#---