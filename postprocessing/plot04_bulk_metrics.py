import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
import pynanigans as pn
import numpy as np
import xarray as xr
from cycler import cycler
from matplotlib import pyplot as plt
from src.aux00_utils import merge_datasets, condense
plt.rcParams["figure.constrained_layout.use"] = True

#+++ Define simulation parameters
simname_bases = ["balanus", "labanus"]

Rossby_numbers = cycler(Ro_b = [0.1])
Froude_numbers = cycler(Fr_b = [1])
L              = cycler(L = [0, 0.05, 0.1, 0.2, 0.4, 0.8])
FWHM           = cycler(FWHM = [500, 500, 500, 500, 500, 500])

resolutions    = cycler(dz = [1])

paramspace = Rossby_numbers * Froude_numbers * (L + FWHM)
configs    = resolutions

runs = paramspace * configs
buffer = 5
#---

#+++ Load and process datasets for both simulations
datasets = {}

for simname_base in simname_bases:
    print(f"\nProcessing {simname_base}...")

    # Load datasets
    aaaa = merge_datasets(runs, base_name=f"aaaa.{simname_base}", verbose=True, add_min_spacings=False,
                          combine_by_coords_kwargs=dict(compat="override", combine_attrs="drop_conflicts", coords="minimal"))
    aaaa = aaaa.reindex(Ro_b = list(reversed(aaaa.Ro_b)))

    # Load aaad datasets to get additional variables
    aaad = merge_datasets(runs, base_name=f"aaad.{simname_base}", verbose=True, add_min_spacings=False, keep_vars=["∭⟨w′b′⟩ₜdV", "∭SPRdV", "U∞∬⟨Ek′⟩ₜdxdz"],
                          combine_by_coords_kwargs=dict(compat="override", combine_attrs="drop_conflicts", coords="minimal"))
    aaad = aaad.reindex(Ro_b = list(reversed(aaad.Ro_b)))

    aaaa = xr.merge([aaaa, aaad], compat="override")

    # Condense buffer variables
    for var in ["ε̄ₚ", "ε̄ₖ"]:
        aaaa = condense(aaaa, [f"∭⁵{var}dV", f"∭¹⁰{var}dV"], f"∭ᵇ{var}dV", dimname="buffer", indices=[5, 10])

    # Create normalized variables
    aaaa["γ"] = aaaa["∭ᵇε̄ₚdV"] / (aaaa["∭ᵇε̄ₚdV"] + aaaa["∭ᵇε̄ₖdV"])
    aaaa["RoFr"] = aaaa.Ro_b * aaaa.Fr_b

    # Normalized dissipation rates
    dtKE_scaling = aaaa.attrs["U∞"]**3 * aaaa.FWHM**2 # Assume ε̄ₖ scales as U^3 / H
    aaaa["ℰₖ"] = aaaa["∭ᵇε̄ₖdV"] / dtKE_scaling
    aaaa["ℰₚ"] = aaaa["∭ᵇε̄ₚdV"] / dtKE_scaling
    aaaa["ℬ"] = -aaaa["∭⟨w′b′⟩ₜdV"] / dtKE_scaling
    aaaa["𝒮"] = aaaa["∭SPRdV"] / dtKE_scaling
    aaaa["𝒯"] = aaaa["U∞∬⟨Ek′⟩ₜdxdz"] / dtKE_scaling
    aaaa["𝒦⁵"] = (aaaa["∭ᵇε̄ₚdV"] / aaaa["N²∞"]) / (aaaa["U∞"] * aaaa.FWHM**2 * aaaa.H**2)

    # Add metadata
    aaaa["ℰₖ"].attrs = dict(long_name=r"TKE dissipation rate $\mathcal{E}_k$")
    aaaa["ℰₚ"].attrs = dict(long_name=r"Buoyancy dissipation rate $\mathcal{E}_p$")
    aaaa["ℬ"].attrs = dict(long_name=r"Turbulent buoyancy flux $\mathcal{B}$")
    aaaa["𝒮"].attrs = dict(long_name=r"Shear production rate $\mathcal{S}$")
    aaaa["𝒯"].attrs = dict(long_name=r"TKE advection out $\mathcal{T}$")
    aaaa["𝒦⁵"].attrs = dict(long_name=r"Buoyancy diffusivity $\mathcal{K}$")

    # Select and store processed data
    aaaa = aaaa.pnsel(x=np.inf, method="nearest") # To get the advection out term
    aaaa = aaaa.sel(dz=0, buffer=buffer, method="nearest").sum("j", keep_attrs=True)
    datasets[simname_base] = aaaa

print("\nData processing complete!")
#---

#+++ Create single plot with all variables
fig, ax = plt.subplots(figsize=(7, 5))

# Variables to plot
variables = ["ℰₖ", "ℰₚ", "ℬ", "𝒮", "𝒯"]
colors = ["blue", "red", "green", "orange", "purple"]

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
from matplotlib.legend import Legend

# Get the automatic legend for variables
var_legend = ax.legend(fontsize=11, loc="upper right", framealpha=0.4)

# Add marker legend
marker_handles = [
    Line2D([0], [0], marker="o", color="gray", linestyle="None", markersize=6, label="Original Balanus"),
    Line2D([0], [0], marker="x", color="gray", linestyle="None", markersize=8, label="90° rotated Balanus")
]
marker_legend = ax.legend(handles=marker_handles, fontsize=11, loc="lower left", framealpha=0.4)

# Add back the variable legend
ax.add_artist(var_legend)

# Set log scale and formatting
ax.set_yscale("log")
ax.set_xlabel("L/W", fontsize=12)
ax.set_ylabel("Value / ($U_\infty^3 L^2$)", fontsize=12)
Ro_b_val = datasets["balanus"].Ro_b.item()
Fr_b_val = datasets["balanus"].Fr_b.item()
ax.set_title(f"Turbulent metrics vs smoothing scale ($Ro_b$={Ro_b_val}, $Fr_b$={Fr_b_val})", fontsize=14)
ax.grid(True, which="both", alpha=0.3)

#+++ Save figure
dz_val = datasets["balanus"].dz.item()
buffer_val = datasets["balanus"].buffer.item()
figure_name = f"../figures/bulk_metrics_comparison_dz{dz_val}_buffer{buffer_val}.pdf"
plt.savefig(figure_name, dpi=300, bbox_inches="tight")
print(f"Figure saved to: {figure_name}")
#---