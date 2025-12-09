import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
import pynanigans as pn
from cycler import cycler
from matplotlib import pyplot as plt
from src.aux00_utils import merge_datasets, condense
from src.aux02_plotting import letterize

#+++ Define simulation parameters for parameter sweep
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
aaaa_sweep = merge_datasets(runs, base_name=f"aaaa.{simname_base}",
                      dirpath="data", verbose=True, add_min_spacings=False,
                      combine_by_coords_kwargs=dict(compat="override", combine_attrs="drop_conflicts", coords="minimal"))
aaaa_sweep = aaaa_sweep.reindex(Ro_b = list(reversed(aaaa_sweep.Ro_b)))
#---

#+++ Process data and create derived variables
# Condense buffer variables
for var in ["ε̄ₚ", "ε̄ₖ"]:
    aaaa_sweep = condense(aaaa_sweep, [f"∭⁵{var}dV", f"∭¹⁰{var}dV"], f"∭ᵇ{var}dV",
                   dimname="buffer", indices=[5, 10])

# Create normalized variables
aaaa_sweep["γ"] = aaaa_sweep["∭ᵇε̄ₚdV"] / (aaaa_sweep["∭ᵇε̄ₚdV"] + aaaa_sweep["∭ᵇε̄ₖdV"])
aaaa_sweep["RoFr"] = aaaa_sweep.Ro_b * aaaa_sweep.Fr_b

# Normalized dissipation rates
norm_factor = aaaa_sweep.attrs["U∞"]**3 * aaaa_sweep.FWHM * aaaa_sweep.H
aaaa_sweep["ℰₖ"] = aaaa_sweep["∭ᵇε̄ₖdV"] / norm_factor
aaaa_sweep["ℰₚ"] = aaaa_sweep["∭ᵇε̄ₚdV"] / norm_factor
aaaa_sweep["γ"] = aaaa_sweep["∭ᵇε̄ₚdV"] / (aaaa_sweep["∭ᵇε̄ₚdV"] + aaaa_sweep["∭ᵇε̄ₖdV"])

# Add metadata
aaaa_sweep["γ"].attrs = dict(long_name=r"Bulk mixing efficiency $\gamma$")
#---

#+++ Create plots
# Define the layout pattern
mosaic = """
ab
cd
"""

# Create the figure and axes using subplot_mosaic
fig, axes = plt.subplot_mosaic(mosaic, figsize=(11, 8))

# Variables to plot (first 3 use buffer=5m, last 2 don"t have buffer dimension)
variables = ["ℰₖ", "ℰₚ", "γ"]

aaaa_sweep = aaaa_sweep.sel(dz=0, buffer=buffer, method="nearest")
# Create plots for each variable
colors = ["red", "blue", "purple", "orange"]  # Generate different colors for each L value

# Store scatter plot handles for shared legend
scatter_handles = []
scatter_labels = []

for i, var_name in zip(axes.keys(), variables):
    data = aaaa_sweep[var_name]
    ax = axes[i]
    for j, L_val in enumerate(aaaa_sweep.L.values):
        subset = aaaa_sweep.sel(L=L_val)
        easy_label = "(rough)" if L_val == 0 else "(smooth)"
        scatter = ax.scatter(x=subset.Slope_Bu, y=subset[var_name], color=colors[j])

        # Only save handles and labels once (from first axis)
        if i == "a":
            scatter_handles.append(scatter)
            scatter_labels.append(f"L/W = {L_val} {easy_label}")

    ax.set_xscale("log")
    ax.set_xlabel("Slope Burger number", fontsize=12)
    ax.set_ylabel(var_name, fontsize=13)
#---

#+++ Add reference lines
import numpy as np
Sb_ref = np.logspace(np.log10(3e-2), np.log10(1e1), 100)

dissip_linear_ref = 2e-2 * Sb_ref
dissip_piecewise_ref = np.maximum(dissip_linear_ref, 2e-2)

mixing_linear_ref = 2e-2 * Sb_ref
mixing_quadratic_ref = 2e-2 * Sb_ref**2

efficiency_ref = 0.5 * Sb_ref

ax = axes["a"]
ax.set_ylabel(f"Normalized dissipation, {ax.get_ylabel()}", fontsize=13)
ax.plot(Sb_ref, dissip_linear_ref, ls="--", lw=5, color="blue", alpha=0.3, label="$\sim S_b$")
ax.plot(Sb_ref, dissip_piecewise_ref, ls="--", lw=5, color="red", alpha=0.3, label=r"$\max(\sim S_b, 2 \times 10^{-2})$")

ax = axes["b"]
ax.set_ylabel(f"Normalized buoyancy mixing, {ax.get_ylabel()}", fontsize=13)
ax.plot(Sb_ref, mixing_linear_ref, ls="--", lw=5, color="gray", alpha=0.5, label="$\sim S_b$")
ax.plot(Sb_ref, mixing_quadratic_ref, ls=":", lw=5, color="gray", alpha=0.5, label="$\sim S_b^2$")

ax = axes["c"]
ax.set_ylabel(f"Bulk mixing efficiency, {ax.get_ylabel()}", fontsize=13)

for ax in (axes["a"], axes["b"]):
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1)
axes["c"].set_ylim(0, 1)

axes["b"].set_yticklabels([])

# Add legends for line plots only (in each panel)
axes["a"].legend(loc="lower right", borderaxespad=0, framealpha=0.7, edgecolor="black", fancybox=False)
axes["b"].legend(loc="lower right", borderaxespad=0, framealpha=0.7, edgecolor="black", fancybox=False)

# Add shared legend for scatter plots at bottom right of figure
fig.legend(scatter_handles, scatter_labels, loc="lower right", bbox_to_anchor=(0.3, 0.28), framealpha=0.7, edgecolor="black", fancybox=False)
#---

#+++ Define simulation parameters for Southern Ocean and labanus comparison
simname_bases = ["balanus", "labanus"]

Rossby_numbers = cycler(Ro_b = [0.1])
Froude_numbers = cycler(Fr_b = [1])
L              = cycler(L = [0, 0.05, 0.1, 0.2, 0.4, 0.8])

paramspace = Rossby_numbers * Froude_numbers * L
configs    = resolutions

runs = paramspace * configs
#---

#+++ Load and process datasets for both simulations
datasets = {}

for simname_base in simname_bases:
    print(f"\nProcessing {simname_base}...")

    # Load datasets
    aaaa_south = merge_datasets(runs, base_name=f"aaaa.{simname_base}", verbose=True, add_min_spacings=False,
                          combine_by_coords_kwargs=dict(compat="override", combine_attrs="drop_conflicts", coords="minimal"))
    aaaa_south = aaaa_south.reindex(Ro_b = list(reversed(aaaa_south.Ro_b)))

    # Condense buffer variables
    for var in ["ε̄ₚ", "ε̄ₖ"]:
        aaaa_south = condense(aaaa_south, [f"∭⁵{var}dV", f"∭¹⁰{var}dV"], f"∭ᵇ{var}dV", dimname="buffer", indices=[5, 10])

    # Create normalized variables
    aaaa_south["γ"] = aaaa_south["∭ᵇε̄ₚdV"] / (aaaa_south["∭ᵇε̄ₚdV"] + aaaa_south["∭ᵇε̄ₖdV"])
    aaaa_south["RoFr"] = aaaa_south.Ro_b * aaaa_south.Fr_b

    # Normalized dissipation rates
    dtKE_scaling = aaaa_south.attrs["U∞"]**3 * aaaa_south.FWHM**2 # Assume ε̄ₖ scales as U^3 / H
    aaaa_south["ℰₖ"] = aaaa_south["∭ᵇε̄ₖdV"] / dtKE_scaling
    aaaa_south["ℰₚ"] = aaaa_south["∭ᵇε̄ₚdV"] / dtKE_scaling

    # Add metadata
    aaaa_south["ℰₖ"].attrs = dict(long_name=r"KE dissipation rate $\mathcal{E}_k$")
    aaaa_south["ℰₚ"].attrs = dict(long_name=r"Buoyancy dissipation rate $\mathcal{E}_p$")

    # Select and store processed data
    aaaa_south = aaaa_south.sel(dz=0, buffer=buffer, method="nearest")
    datasets[simname_base] = aaaa_south

print("\nData processing complete!")
#---

#+++ Plot fourth panel with both ℰₖ and ℰₚ
ax = axes["d"]

variables = ["ℰₖ", "ℰₚ"]
colors = ["green", "#E377C2"]

# Marker styles for each simulation
markers = {"balanus": "o", "labanus": "x"}
marker_sizes = {"balanus": 6, "labanus": 8}

# Plot each variable for both simulations
for var_name, color in zip(variables, colors):
    for i, (simname, aaaa) in enumerate(datasets.items()):
        variable_da = aaaa[var_name]
        label = variable_da.attrs["long_name"] if i == 0 else None  # Only label once per variable
        alpha = 0.5 if simname == "labanus" else 1.0
        variable_da.plot.scatter(ax=ax, x="L", label=label, color=color,
                                 marker=markers[simname], s=marker_sizes[simname]**2, alpha=alpha)

# Create custom legend with both variables and markers
from matplotlib.lines import Line2D

# Get the automatic legend for variables
var_legend = ax.legend(fontsize=11, framealpha=0.4, loc="upper right", bbox_to_anchor=(0.98, 0.98))

# Add marker legend
marker_handles = [
    Line2D([0], [0], marker="o", color="gray", linestyle="None", markersize=6, label="Original Balanus"),
    Line2D([0], [0], marker="x", color="gray", linestyle="None", markersize=8, label="90° rotated Balanus")
]
marker_legend = ax.legend(handles=marker_handles, fontsize=11, framealpha=0.4, loc="upper right", bbox_to_anchor=(0.98, 0.78))

# Add back the variable legend
ax.add_artist(var_legend)

# Set log scale and formatting
ax.set_yscale("log")
ax.set_xlabel("$L/W$", fontsize=12)
ax.set_ylabel("", fontsize=12)

dataset = datasets[list(datasets.keys())[0]]

Ro_b_val = dataset.Ro_b.item()
Fr_b_val = dataset.Fr_b.item()
delta_val = dataset.H.item() / dataset.FWHM.item()
Slope_Bu_SO = dataset.Slope_Bu.mean().item()
ax.set_ylabel(f"ℰₖ, ℰₚ for $S_b$ = {Slope_Bu_SO:.1f}", fontsize=13)
ax.set_title("")
#---

#+++ Prettify
for ax in [axes["a"], axes["b"], axes["c"]]:
    ax.axvline(x=Slope_Bu_SO, color="gray", linestyle="--", linewidth=2, alpha=0.5)
for ax in axes.values():
    ax.grid(True, which="both", alpha=0.3)
letterize(axes.values(), x=0.11, y=0.92, fontsize=13, bbox=dict(boxstyle="square", facecolor="white", alpha=0.4))
#---

#+++ Save figure
figure_name = f"../figures/paramsweep_bulk_metrics_{simname_bases[0]}_dz{aaaa_sweep.dz.item()}_buffer{aaaa_sweep.buffer.item()}.pdf"
plt.savefig(figure_name, dpi=300, bbox_inches="tight")
print(f"Figure saved to: {figure_name}")
#---
