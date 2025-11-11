import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
import pynanigans as pn
from cycler import cycler
from matplotlib import pyplot as plt
from src.aux00_utils import merge_datasets, condense
from src.aux02_plotting import letterize

#+++ Define simulation parameters
simname_base = "balanus"

Rossby_numbers = cycler(Ro_b = [0.05, 0.1, 0.2, 0.5])
Froude_numbers = cycler(Fr_b = [0.05, 0.08, 0.3, 1, 2])
L              = cycler(L = [0, 0.8])

resolutions    = cycler(dz = [1])
T_adv_spinups  = cycler(T_adv_spinup = [12])

paramspace = Rossby_numbers * Froude_numbers * L
configs    = resolutions  * T_adv_spinups

runs = paramspace * configs
buffer = 5
#---

#+++ Load datasets
aaaa = merge_datasets(runs, base_name=f"aaaa.{simname_base}",
                      dirpath="../paramsweep_postprocessing/data", verbose=True, add_min_spacings=False,
                      combine_by_coords_kwargs=dict(compat="override", combine_attrs="drop_conflicts", coords="minimal"))
aaaa = aaaa.reindex(Ro_b = list(reversed(aaaa.Ro_b)))
#---

#+++ Process data and create derived variables
# Condense buffer variables
for var in ["ε̄ₚ", "ε̄ₖ"]:
    aaaa = condense(aaaa, [f"∭⁵{var}dV", f"∭¹⁰{var}dV"], f"∭ᵇ{var}dV",
                   dimname="buffer", indices=[5, 10])

# Create normalized variables
aaaa["γ"] = aaaa["∭ᵇε̄ₚdV"] / (aaaa["∭ᵇε̄ₚdV"] + aaaa["∭ᵇε̄ₖdV"])
aaaa["RoFr"] = aaaa.Ro_b * aaaa.Fr_b

# Normalized dissipation rates
norm_factor = aaaa.attrs["U∞"]**3 * aaaa.FWHM * aaaa.H
aaaa["ℰₖ"] = aaaa["∭ᵇε̄ₖdV"] / norm_factor
aaaa["ℰₚ"] = aaaa["∭ᵇε̄ₚdV"] / norm_factor
aaaa["𝒦⁵"] = (aaaa["∭ᵇε̄ₚdV"] / aaaa["N²∞"]) / (aaaa["U∞"] * aaaa.FWHM**2 * aaaa.H**2)
aaaa["γ"] = aaaa["∭ᵇε̄ₚdV"] / (aaaa["∭ᵇε̄ₚdV"] + aaaa["∭ᵇε̄ₖdV"])

# Add metadata
aaaa["𝒦⁵"].attrs = dict(long_name=r"Norm buoyancy diffusivity $\mathcal{K}$")
aaaa["γ"].attrs = dict(long_name=r"Bulk mixing efficiency $\gamma$")
#---

#+++ Create plots
# Define the layout pattern
mosaic = """
aabb
.cc.
"""

# Create the figure and axes using subplot_mosaic
fig, axes = plt.subplot_mosaic(mosaic, figsize=(10, 8), gridspec_kw=dict(wspace=-1))

# Variables to plot (first 3 use buffer=5m, last 2 don"t have buffer dimension)
variables = ["ℰₖ", "ℰₚ", "γ"]

aaaa = aaaa.sel(dz=0, buffer=buffer, method="nearest")
# Create plots for each variable
colors = ["red", "blue", "purple", "orange"]  # Generate different colors for each L value

for i, var_name in zip(axes.keys(), variables):
    data = aaaa[var_name]
    ax = axes[i]
    for j, L_val in enumerate(aaaa.L.values):
        subset = aaaa.sel(L=L_val)
        ax.scatter(x=subset.Slope_Bu, y=subset[var_name], color=colors[j], label=f"L/W = {L_val}")

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.grid(True)
    ax.set_xlabel("Slope Burger number")
    ax.set_ylabel(var_name)
#---

#+++ Add reference lines
import numpy as np
Sb_ref = np.logspace(np.log10(3e-2), np.log10(1e1), 100)

dissip_linear_ref = 2e-2 * Sb_ref
dissip_piecewise_ref = np.maximum(dissip_linear_ref, 2e-2)

mixing_linear_ref = 2e-2 * Sb_ref
mixing_quadratic_ref = 2e-2 * Sb_ref**2

efficiency_ref = 0.5 * Sb_ref

axes["a"].set_title("Normalized dissipation")
axes["a"].plot(Sb_ref, dissip_linear_ref, ls="--", lw=5, color="blue", alpha=0.3, label="$\sim S_b$")
axes["a"].plot(Sb_ref, dissip_piecewise_ref, ls="--", lw=5, color="red", alpha=0.3, label=r"$\max(\sim S_b, 2 \times 10^{-2})$")

axes["b"].set_title("Normalized buoyancy mixing")
axes["b"].set_yticklabels([])
axes["b"].plot(Sb_ref, mixing_linear_ref, ls="--", lw=5, color="gray", alpha=0.5, label="$\sim S_b$")
axes["b"].plot(Sb_ref, mixing_quadratic_ref, ls=":", lw=5, color="gray", alpha=0.5, label="$\sim S_b^2$")

axes["c"].set_title("Bulk mixing efficiency")
axes["c"].plot(Sb_ref, efficiency_ref, ls="--", lw=5, color="gray", alpha=0.5, label="$\sim S_b$")

for ax in (axes["a"], axes["b"]):
    ax.set_ylim(1e-5, 1)
axes["c"].set_ylim(5e-3, 1)

for ax in axes.values():
    ax.legend(loc="lower right", borderaxespad=0, framealpha=0.7, edgecolor="black", fancybox=False)

letterize(axes.values(), x=0.05, y=0.92, fontsize=11, bbox=dict(boxstyle="square", facecolor="white", alpha=0.4))
#---


#+++ Save figure
figure_name = f"../figures/paramsweep_bulk_metrics_{simname_base}_dz{aaaa.dz.item()}_buffer{aaaa.buffer.item()}.pdf"
plt.savefig(figure_name, dpi=300, bbox_inches="tight")
print(f"Figure saved to: {figure_name}")
#---
