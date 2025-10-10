import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
sys.path.append("../postprocessing")
import pynanigans as pn
from cycler import cycler
from matplotlib import pyplot as plt
from src.aux00_utils import merge_datasets, condense
from IPython import embed
plt.rcParams["figure.constrained_layout.use"] = True

#+++ Define simulation parameters
simname_base = "seamount"

Rossby_numbers = cycler(Ro_h = [0.05, 0.1, 0.2, 0.5])
Froude_numbers = cycler(Fr_h = [0.02, 0.08, 0.3, 1])
L              = cycler(L = [0, 0.8])

resolutions    = cycler(dz = [1])
T_advective_spinups = cycler(T_advective_spinup = [12])

paramspace = Rossby_numbers * Froude_numbers * L
configs    = resolutions  * T_advective_spinups

runs = paramspace * configs
#---

#+++ Load datasets
aaaa = merge_datasets(runs, base_name=f"aaaa.{simname_base}", verbose=True, add_min_spacings=False)
aaaa = aaaa.reindex(Ro_h = list(reversed(aaaa.Ro_h)))
#---

#+++ Process data and create derived variables
# Condense buffer variables
for var in ["ε̄ₚ", "ε̄ₖ"]:
    aaaa = condense(aaaa, [f"∭⁵{var}dV", f"∭¹⁰{var}dV"], f"∭ᵇ{var}dV",
                   dimname="buffer", indices=[5, 10])

# Create normalized variables
aaaa["γ"] = aaaa["∭ᵇε̄ₚdV"] / (aaaa["∭ᵇε̄ₚdV"] + aaaa["∭ᵇε̄ₖdV"])
aaaa["RoFr"] = aaaa.Ro_h * aaaa.Fr_h

# Normalized dissipation rates
norm_factor = aaaa.attrs["U∞"]**3 * aaaa.FWHM * aaaa.H
aaaa["ℰₖ"] = aaaa["∭ᵇε̄ₖdV"] / norm_factor
aaaa["ℰₚ"] = aaaa["∭ᵇε̄ₚdV"] / norm_factor
aaaa["𝒦⁵"] = (aaaa["∭ᵇε̄ₚdV"] / aaaa["N²∞"]) / (aaaa["U∞"] * aaaa.FWHM**2 * aaaa.H**2)

# Add metadata
aaaa["𝒦⁵"].attrs = dict(long_name=r"Norm buoyancy diffusivity $\mathcal{K}$")
#---

#+++ Create plots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharey=True)
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# Variables to plot (first 3 use buffer=5m, last 2 don"t have buffer dimension)
variables = ["ℰₖ", "ℰₚ"]

aaaa = aaaa.sel(dz=0, buffer=10, method="nearest")
# Create plots for each variable
colors = ["red", "blue", "purple", "orange"]  # Generate different colors for each L value

for i, var_name in enumerate(variables):
    data = aaaa[var_name]
    ax = axes[i]
    for j, L_val in enumerate(aaaa.L.values):
        subset = aaaa.sel(L=L_val)
        ax.scatter(x=subset.Slope_Bu, y=subset[var_name], color=colors[j], label=f"L = {L_val} m")

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_title(f"{var_name}")
    ax.grid(True)
    ax.legend()
    ax.set_xlabel("Slope Burger number")
    ax.set_ylabel(var_name)
#---

#+++ Add reference lines
import numpy as np
x_ref = np.logspace(np.log10(2e-1), np.log10(1e1), 100)
y_ref = 1e-2 * x_ref

for ax in axes:
    ax.plot(x_ref, y_ref, '--k', alpha=0.5)
#---


#+++ Save figure
figure_name = f"../figures/paramsweep_bulk_metrics_{simname_base}.png"
plt.savefig(figure_name, dpi=300, bbox_inches="tight")
print(f"Figure saved to: {figure_name}")
#---