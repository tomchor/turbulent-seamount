import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
import pynanigans as pn
from cycler import cycler
from matplotlib import pyplot as plt
from src.aux00_utils import merge_datasets, condense
plt.rcParams["figure.constrained_layout.use"] = True

#+++ Define simulation parameters
simname_base = "seamount"

Froude_numbers = cycler(Fr_h = [1])
L              = cycler(L = [0, 0.05, 0.1, 0.2, 0.4, 0.8,
                             0.8, 0.8])
FWHM           = cycler(FWHM = [500, 500, 500, 500, 500, 500,
                                200, 100])

resolutions    = cycler(dz = [4, 2])

paramspace = Rossby_numbers * Froude_numbers * (L + FWHM)
configs    = resolutions

runs = paramspace * configs
#---

aaaa = merge_datasets(runs, base_name=f"aaaa.{simname_base}", verbose=True, add_min_spacings=False)
aaaa = aaaa.reindex(Ro_h = list(reversed(aaaa.Ro_h)))

# Load turbstats datasets to get additional variables
turbstats = merge_datasets(runs, base_name=f"turbstats.{simname_base}", verbose=True, add_min_spacings=False)
turbstats = turbstats.reindex(Ro_h = list(reversed(turbstats.Ro_h)))

# fit_filename = f'data/bathymetry_powerlaw_fits_{simname_base}.nc'
# ds_fit = xr.open_dataset(fit_filename).sel(L=slice(0, 400))
# aaaa = xr.merge([aaaa, ds_fit])

#+++ Define new variables
#+++ Condense buffers
distances = [5, 10]
aaaa = condense(aaaa, ["∭⁵ε̄ₚdV", "∭¹⁰ε̄ₚdV"], "∭ᵇε̄ₚdV", dimname="buffer", indices=distances)
aaaa = condense(aaaa, ["∭⁵ε̄ₖdV", "∭¹⁰ε̄ₖdV"], "∭ᵇε̄ₖdV", dimname="buffer", indices=distances)
#---

aaaa["γ"] = aaaa["∭ᵇε̄ₚdV"] / (aaaa["∭ᵇε̄ₚdV"] + aaaa["∭ᵇε̄ₖdV"])

aaaa["RoFr"] = aaaa.Ro_h * aaaa.Fr_h

# hor_scale = 1/aaaa.transition_wavenumber
hor_scale = aaaa.FWHM
aaaa["ℰₖ"] = aaaa["∭ᵇε̄ₖdV"] / (aaaa.attrs["U∞"]**3 * aaaa.FWHM**2 * aaaa.H / hor_scale)
aaaa["ℰₚ"] = aaaa["∭ᵇε̄ₚdV"] / (aaaa.attrs["U∞"]**3 * aaaa.FWHM**2 * aaaa.H / hor_scale)

aaaa["𝒦⁵"] = (aaaa["∭ᵇε̄ₚdV"] / aaaa["N²∞"]) / (aaaa["U∞"] * aaaa.FWHM**2 * aaaa.H**2)
#---

#+++ Make it legible
aaaa["𝒦⁵"].attrs = dict(long_name=r"Norm buoyancy diffusivity $\mathcal{K}$")
#---

# aaaa = aaaa.where(aaaa.Slope_Bu==0.1, drop=True).squeeze()

#+++ Create 2x2 subplot grid (using 3 panels for 5m buffer results)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# Flatten axes array for easier indexing
axes_flat = axes.flatten()

#+++ Plot ℰₖ for buffer=5m (top-left)
data_5m_k = aaaa["ℰₖ"].sel(dz=0, method="nearest").sel(buffer=5)
for fwhm_val in data_5m_k.FWHM.values:
    subset = data_5m_k.sel(FWHM=fwhm_val)
    axes_flat[0].scatter(subset.L, subset.values, label=f'FWHM={fwhm_val}', alpha=0.7)
axes_flat[0].set_yscale('log')
axes_flat[0].set_xlabel('L')
axes_flat[0].set_ylabel('ℰₖ')
axes_flat[0].set_title('ℰₖ (buffer=5m)')
axes_flat[0].grid(True)
axes_flat[0].axvline(x=1, color="black", linestyle="--", alpha=0.5)
axes_flat[0].legend()

#+++ Plot ℰₚ for buffer=5m (top-right)
data_5m_p = aaaa["ℰₚ"].sel(dz=0, method="nearest").sel(buffer=5)
for fwhm_val in data_5m_p.FWHM.values:
    subset = data_5m_p.sel(FWHM=fwhm_val)
    axes_flat[1].scatter(subset.L, subset.values, label=f'FWHM={fwhm_val}', alpha=0.7)
axes_flat[1].set_yscale('log')
axes_flat[1].set_xlabel('L')
axes_flat[1].set_ylabel('ℰₚ')
axes_flat[1].set_title('ℰₚ (buffer=5m)')
axes_flat[1].grid(True)
axes_flat[1].axvline(x=1, color="black", linestyle="--", alpha=0.5)
axes_flat[1].legend()

#+++ Plot 𝒦⁵ for buffer=5m (bottom-left)
data_5m_kappa = aaaa["𝒦⁵"].sel(dz=0, method="nearest").sel(buffer=5)
for fwhm_val in data_5m_kappa.FWHM.values:
    subset = data_5m_kappa.sel(FWHM=fwhm_val)
    axes_flat[2].scatter(subset.L, subset.values, label=f'FWHM={fwhm_val}', alpha=0.7)
axes_flat[2].set_yscale('log')
axes_flat[2].set_xlabel('L')
axes_flat[2].set_ylabel('𝒦⁵')
axes_flat[2].set_title('𝒦⁵ (buffer=5m)')
axes_flat[2].grid(True)
axes_flat[2].axvline(x=1, color="black", linestyle="--", alpha=0.5)
axes_flat[2].legend()

#+++ Hide the unused subplot (bottom-right)
axes_flat[3].set_visible(False)

plt.tight_layout()
