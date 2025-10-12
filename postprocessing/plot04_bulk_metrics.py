import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
import pynanigans as pn
import xarray as xr
from cycler import cycler
from matplotlib import pyplot as plt
from src.aux00_utils import merge_datasets, condense
plt.rcParams["figure.constrained_layout.use"] = True

#+++ Define simulation parameters
simname_base = "seamount"

Rossby_numbers = cycler(Ro_h = [0.1])
Froude_numbers = cycler(Fr_h = [1])
L              = cycler(L = [0, 0.05, 0.1, 0.2, 0.4, 0.8])
FWHM           = cycler(FWHM = [500, 500, 500, 500, 500, 500])

resolutions    = cycler(dz = [2, 1])

paramspace = Rossby_numbers * Froude_numbers * (L + FWHM)
configs    = resolutions

runs = paramspace * configs
#---

#+++ Load datasets
aaaa = merge_datasets(runs, base_name=f"aaaa.{simname_base}", verbose=True, add_min_spacings=False)
aaaa = aaaa.reindex(Ro_h = list(reversed(aaaa.Ro_h)))

# Load aaad datasets to get additional variables
aaad = merge_datasets(runs, base_name=f"aaad.{simname_base}", verbose=True, add_min_spacings=False, keep_vars=["∭⟨w′b′⟩ₜdV", "∭SPRdV"])
aaad = aaad.reindex(Ro_h = list(reversed(aaad.Ro_h)))

aaaa = xr.merge([aaaa, aaad], compat="override")
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

#+++ Helper function to create scatter plot
def plot_variable(ax, data, var_name):
    """Create a scatter plot for a given variable on the specified axis"""
    for fwhm_val in data.FWHM.values:
        subset = data.sel(FWHM=fwhm_val)
        ax.scatter(subset.L, subset.values, label=f"FWHM={fwhm_val}", alpha=0.7)

    # Use symlog scale for w"b" variable (can be positive or negative)
    if var_name == "∭⟨w′b′⟩ₜdV":
        ax.set_yscale("symlog", linthresh=1e-6)
    else:
        ax.set_yscale("log")

    ax.set_xlabel("L")
    ax.set_ylabel(var_name)
    ax.set_title(f"{var_name}")
    ax.grid(True)
    ax.axvline(x=1, color="black", linestyle="--", alpha=0.5)
    ax.legend()

#+++ Create plots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# Variables to plot (first 3 use buffer=5m, last 2 don"t have buffer dimension)
variables = ["ℰₖ", "ℰₚ", "∭⟨w′b′⟩ₜdV", "∭SPRdV"]

aaaa = aaaa.sel(dz=0, buffer=5, method="nearest").sum("j")
# Create plots for each variable
for i, var_name in enumerate(variables):
    data = aaaa[var_name]
    plot_variable(axes.flat[i], data, var_name)

#+++ Save figure
figure_name = f"../figures/bulk_metrics_{simname_base}_dz{aaaa.dz.item()}m_buffer{aaaa.buffer.item()}m.png"
plt.savefig(figure_name, dpi=300, bbox_inches="tight")
print(f"Figure saved to: {figure_name}")
#---