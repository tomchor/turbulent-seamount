import numpy as np
import xarray as xr
from .aux00_utils import normalize_unicode_names_in_dataset
from scipy.optimize import curve_fit
π = np.pi


#+++ Define filtering functions
def filter_1d(da, filter_size, dim="x_caa", kernel="gaussian", spacing=None, min_distance=0,
              optimize=True, verbose=False, keep_dim_attrs=True):
    x2 = da[dim].rename({ dim : "x2"}).chunk()

    if spacing is None:
        raise(ValueError(f"Need spacing at grid points along dimension {dim}"))

    if kernel == "tophat":
        aux = (abs(x2 - da[dim]) < filter_size/2).astype(float)
    elif kernel == "gaussian":
        aux = np.exp(-((x2 - da[dim]) / (filter_size/2))**2)
    else:
        raise(TypeError("Invalid kernel name"))

    weights = (aux / aux.integrate(dim)).chunk("auto")
    if verbose: print(weights)

    if optimize:
        da_f = xr.dot(da, weights, spacing, dims=dim, optimize=True).rename(x2=dim)
    else:
        da_f = (da * weights).integrate(dim).rename(x2=dim)

    if min_distance: # Exclude edges (easier to do it after the convolution I think)
        da_f = da_f.where((da_f[dim]     - da_f[dim][0] >= min_distance) & \
                          (da_f[dim][-1] - da_f[dim]    >= min_distance), other=np.nan)
    if keep_dim_attrs:
        for dimname in da_f.dims:
            da_f[dimname].attrs = da[dimname].attrs

    return da_f

def coarsen(da, filter_size, dims=["x_caa", "y_aca"], kernel="gaussian",
            spacings=[None, None], min_distance=0, optimize=True, verbose=False):
    result = da
    for dim, spacing in zip(dims, spacings):
        result = filter_1d(result, filter_size, dim=dim, kernel=kernel, spacing=spacing, min_distance=min_distance, optimize=optimize, verbose=verbose)
    return result
#---

#+++ Calculate filtered PV
def calculate_filtered_PV(ds, scale_meters = 5, condense_tensors=False, indices = [1,2,3], cleanup=False, normalize_unicode=True):
    from .aux00_utils import condense
    if condense_tensors:
        ds = condense(ds, ["∂u∂x", "∂v∂x", "∂w∂x"], "∂₁uᵢ", dimname="i", indices=indices)
        ds = condense(ds, ["∂u∂y", "∂v∂y", "∂w∂y"], "∂₂uᵢ", dimname="i", indices=indices)
        ds = condense(ds, ["∂u∂z", "∂v∂z", "∂w∂z"], "∂₃uᵢ", dimname="i", indices=indices)
        ds = condense(ds, ["∂₁uᵢ", "∂₂uᵢ", "∂₃uᵢ"], "∂ⱼuᵢ", dimname="j", indices=indices)
        ds = condense(ds, ["dbdx", "dbdy", "dbdz"], "∂ⱼb",  dimname="j", indices=indices)

    ds["∂ⱼũᵢ"] = coarsen(ds["∂ⱼuᵢ"], scale_meters, dims=["x_caa", "y_aca"], spacings=[ds["Δx_caa"], ds["Δy_aca"]], optimize=True)
    ds["∂ⱼb̃"]  = coarsen(ds["∂ⱼb"],  scale_meters, dims=["x_caa", "y_aca"], spacings=[ds["Δx_caa"], ds["Δy_aca"]], optimize=True)

    ω_x = ds["∂ⱼũᵢ"].sel(i=3, j=2) - ds["∂ⱼũᵢ"].sel(i=2, j=3)
    ω_y = ds["∂ⱼũᵢ"].sel(i=1, j=3) - ds["∂ⱼũᵢ"].sel(i=3, j=1)
    ω_z = ds["∂ⱼũᵢ"].sel(i=2, j=1) - ds["∂ⱼũᵢ"].sel(i=1, j=2)

    ds["q̃x"] = ω_x * ds["∂ⱼb̃"].sel(j=1)
    ds["q̃y"] = ω_y * ds["∂ⱼb̃"].sel(j=2)
    ds["q̃z"] = ω_z * ds["∂ⱼb̃"].sel(j=3) + ds["f₀"] * ds["∂ⱼb̃"].sel(j=3)

    ds = condense(ds, ["q̃x", "q̃y", "q̃z"], "q̃ᵢ", dimname="i", indices=indices)
    ds["q̃"] = ds["q̃ᵢ"].sum("i")

    if cleanup:
        ds = ds.drop_vars(["∂ⱼũᵢ", "∂ⱼb̃", "q̃ᵢ",])

    # Normalize Unicode names in one place
    ds = normalize_unicode_names_in_dataset(ds, normalize_unicode)
    return ds
#---

#+++ Get important masks
def get_topography_masks(ds, buffers_in_meters=[0, 5, 10, 30], get_buffered_volumes=True, load=False):
    """ Get some important masks for the headland set-up"""

    #+++ Easy stuff first
    ds["land_mask"]  = (ds["Δx_caa"] == 0)
    ds["water_mask"] = np.logical_not(ds.land_mask)
    #---

    #+++ Get the distances from topography
    squared_distances = []

    if "x_caa" in ds.reset_coords().coords.keys():
        x_boundary_locations = ds.x_caa.where(ds.land_mask.astype(float).diff("x_caa")).max("x_caa")
        x_boundary_locations = x_boundary_locations.where(np.isfinite(x_boundary_locations), other=400)
        x_squared_distances = (ds.x_caa - x_boundary_locations)**2
        squared_distances.append(x_squared_distances)

    if "y_aca" in ds.reset_coords().coords.keys():
        y_boundary_locations_north = ds.y_aca.where(ds.land_mask.astype(float).diff("y_aca")).max("y_aca")
        y_boundary_locations_north = y_boundary_locations_north.where(np.isfinite(y_boundary_locations_north), other=0)

        y_boundary_locations_south = ds.y_aca.where(ds.land_mask.astype(float).diff("y_aca")).min("y_aca")
        y_boundary_locations_south = y_boundary_locations_south.where(np.isfinite(y_boundary_locations_south), other=0)

        y_squared_distances = np.sqrt(xr.concat([(ds.y_aca - y_boundary_locations_south)**2, (ds.y_aca - y_boundary_locations_north)**2], dim="aux").min("aux"))
        squared_distances.append(y_squared_distances)

    if "z_aac" in ds.reset_coords().coords.keys():
        pass
        #raise(ValueError("z direction not yet implemented"))

    ds["distance_from_boundary"] = np.sqrt(xr.concat(squared_distances, dim="aux").sum("aux")).where(ds.water_mask, other=0)
    ds["distance_from_boundary"] = xr.concat([np.sqrt(x_squared_distances), ds.distance_from_boundary], dim="aux").min("aux").where(ds.water_mask, other=0)
    #---

    #+++ Get buffered masks
    filter_scales = xr.DataArray(data=np.array(buffers_in_meters), dims=["buffer"], coords=dict(buffer=buffers_in_meters))
    filter_scales.attrs = dict(units="meters")
    ds["water_mask_buffered"] = (ds.distance_from_boundary > filter_scales)

    if get_buffered_volumes:
        dV = ds["Δx_caa"] * ds["Δy_aca"] * ds["Δz_aac"]
        ds["dV_water_mask_buffered"] = dV.where(ds.water_mask_buffered, other=0.0)
    #---

    if load:
        from dask.diagnostics.progress import ProgressBar
        with ProgressBar(minimum=3, dt=1):
            ds.water_mask_buffered.load()
            ds.dV_water_mask_buffered.load()
    return ds
#---

#+++ Convenience functions
def get_relevant_attrs(ds):
    """
    Get what are deemed relevant attrs from ds.attrs and return them
    as a dataset
    """
    wantedkeys = ["Lx", "Ly", "Lz", "N2_inf", "z_0", "interval",
                  "T_inertial", "Nx", "Ny", "Nz", "date", "b_0",
                  "Oceananigans", "y_0", "ν_eddy", "f_0", "LES",
                  "Ro_s", "Fr_s", "δ", "Rz", "Re_eddy", "L", "H"]
    attrs = dict((k, v) for k,v in ds.attrs.items() if k in wantedkeys)
    return attrs
#---

#+++ Bathymetry
def seamount_curve(x, y, p):
    """ Calculates the seamount curve """
    return p.H * np.exp(-(x/p.L)**2 - (y/p.L)**2)
#---

#+++ Temporal averaging functions
def temporal_average(ds, rename_dict=None, normalize_unicode=True):
    """
    Apply temporal averaging and rename variables. Both the overbar and the angles notation mean a time average.
    That is, ū and ⟨u⟩ₜ are exactly the same.
    """
    if rename_dict is None:
        rename_dict = {"uᵢ"      : "ūᵢ",
                       "b"       : "b̄",
                       "uⱼuᵢ"    : "⟨uⱼuᵢ⟩ₜ",
                       "uᵢuᵢ"    : "⟨uᵢuᵢ⟩ₜ",
                       "wb"      : "⟨wb⟩ₜ",
                       "∂ⱼuᵢ"    : "∂ⱼūᵢ",
                       "εₖ"      : "ε̄ₖ",
                       "εₚ"      : "ε̄ₚ",
                       "εₛ"      : "ε̄ₛ",
                       "ν"       : "ν̄",
                       "κ"       : "κ̄",
                       "PV"      : "q̄",
                       "Ri"      : "R̄i",
                       "Ro"      : "R̄o",
                       "ω_y"     : "ω̄_y",
                       "∭⁵εₖdV"  : "∭⁵ε̄ₖdV",
                       "∭⁵εₚdV"  : "∭⁵ε̄ₚdV",
                       "∭⁵εₛdV"  : "∭⁵ε̄ₛdV",
                       "∭¹⁰εₖdV" : "∭¹⁰ε̄ₖdV",
                       "∭¹⁰εₚdV" : "∭¹⁰ε̄ₚdV",
                       "∭¹⁰εₛdV" : "∭¹⁰ε̄ₛdV",
                       "∭²⁰εₖdV" : "∭²⁰ε̄ₖdV",
                       "∭²⁰εₚdV" : "∭²⁰ε̄ₚdV",
                       "∭²⁰εₛdV" : "∭²⁰ε̄ₛdV",
                       "∭εₛdV"   : "∭ε̄ₛdV",
                       "∫εₖdx"    : "∫ε̄ₖdx",
                       "∫εₚdx"    : "∫ε̄ₚdx",
                       "∫εₛdx"    : "∫ε̄ₛdx",
                       }

    # Filter rename_dict to only include keys that exist in the dataset
    filtered_rename_dict = {key: value for key, value in rename_dict.items() if key in ds.variables}
    ds = ds.mean("time", keep_attrs=True).rename(filtered_rename_dict)

    # Normalize Unicode names in one place
    ds = normalize_unicode_names_in_dataset(ds, normalize_unicode)
    return ds
#---


#+++ Spectral analysis functions
def piecewise_powerlaw(k, amp, alpha, k_transition):
    """
    Piecewise function: power law k^alpha for k < k_transition, constant for k >= k_transition,
    with continuity at k_transition
    """
    # Compute constant to ensure continuity at k_transition
    constant = amp * k_transition**alpha
    return np.where(k < k_transition,
                    amp * k**alpha,
                    constant)

def piecewise_powerlaw_log(log_k, log_amp, alpha, log_k_transition):
    """
    Piecewise function in log-log space:
    log(S) = log_amp + alpha * log_k for log_k < log_k_transition
    log(S) = log_amp + alpha * log_k_transition for log_k >= log_k_transition
    """
    # Constant part in log space
    log_constant = log_amp + alpha * log_k_transition
    return np.where(log_k < log_k_transition,
                    log_amp + alpha * log_k,
                    log_constant)

def fit_piecewise_powerlaw_spectrum(k, S, initial_k_transition=None, alpha_0=-2.0, debug=False, mask_arrays=False):
    """
    Fit a piecewise power law to a spectrum in log-log space.

    Parameters:
    -----------
    k : array_like
        Wavenumber array
    S : array_like
        Spectrum values
    initial_k_transition : float, optional
        Initial guess for transition wavenumber. If None, uses median of k range.
    alpha_0 : float, optional
        Initial guess for power law exponent. Default -2.0.
    debug : bool, optional
        If True, create debug plots. Default False.

    Returns:
    --------
    dict
        Dictionary containing fit results:
        - "amp": amplitude parameter
        - "alpha": power law exponent
        - "k_transition": transition wavenumber
        - "fit_success": whether fit succeeded
        - "fitted_spectrum": fitted spectrum values on original k grid
    """

    #+++ Remove zero or negative values for log fitting
    if mask_arrays:
        valid_mask = (S > 0) & (k > 0)
        k_fit = k[valid_mask]
        S_fit = S[valid_mask]
    else:
        k_fit = k
        S_fit = S
    #---

    if len(k_fit) < 3:
        return {
            "amp": np.nan,
            "alpha": np.nan,
            "k_transition": np.nan,
            "fit_success": False,
            "fitted_spectrum": np.nan * k
        }

    # Convert to log space
    log_k = np.log(k_fit)
    log_S = np.log(S_fit)

    # Initial parameter guesses in log space
    if initial_k_transition is None:
        initial_k_transition = np.median(k_fit)

    log_amp_0 = np.log(S_fit[0]) - alpha_0 * log_k[0]  # Consistent with power law at first point
    log_k_trans_0 = np.log(initial_k_transition)
    p0 = [log_amp_0, alpha_0, log_k_trans_0]

    # Bounds in log space: log_amp (any), alpha<0 (decay), log_k_transition (within data range)
    bounds = ([-np.inf, -10, log_k[0]],
              [np.inf, 0, log_k[-1]])

    try:
        # Fit the piecewise function in log-log space
        popt_log, pcov_log = curve_fit(piecewise_powerlaw_log, log_k, log_S,
                                       p0=p0, bounds=bounds, method="dogbox")

        # Convert back to linear space parameters
        log_amp, alpha, log_k_transition = popt_log
        amp = np.exp(log_amp)
        k_transition = np.exp(log_k_transition)

        # Generate fitted spectrum for plotting on original k grid
        fitted_spectrum = piecewise_powerlaw(k, amp, alpha, k_transition)

        # Debug plotting
        if debug:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.loglog(k_fit, S_fit, "k-", label="Original (valid points)")
            plt.loglog(k, fitted_spectrum, "r--", label="Fitted (full grid)")
            plt.axvline(k_transition, color="g", linestyle=":", label="Transition")
            plt.legend()
            plt.xlabel("Wavenumber k")
            plt.ylabel("Spectrum S")
            plt.title("Piecewise Power Law Fit (Log-log space)")
            plt.show()

        return {
            "amp": amp,
            "alpha": alpha,
            "k_transition": k_transition,
            "fit_success": True,
            "fitted_spectrum": fitted_spectrum
        }

    except Exception as e:
        if debug:
            print(f"Fit failed: {e}")

        return {
            "amp": np.nan,
            "alpha": np.nan,
            "k_transition": np.nan,
            "fit_success": False,
            "fitted_spectrum": np.nan * k
        }
#---

#+++ Turbulence calculations
def get_turbulent_Reynolds_stress_tensor(ds, normalize_unicode=True):
    """Calculate turbulent Reynolds stress tensor"""
    ds["ūⱼūᵢ"]      = ds["ūᵢ"] * ds["ūᵢ"].rename(i="j")
    ds["⟨u′ⱼu′ᵢ⟩ₜ"] = ds["⟨uⱼuᵢ⟩ₜ"] - ds["ūⱼūᵢ"]

    # Normalize Unicode names in one place
    ds = normalize_unicode_names_in_dataset(ds, normalize_unicode)
    return ds

def get_turbulent_Reynolds_stress_tensor_diagonal(ds, normalize_unicode=True):
    """Calculate turbulent Reynolds stress tensor diagonal"""
    ds["ūᵢūᵢ"]      = ds["ūᵢ"] * ds["ūᵢ"]
    ds["⟨u′ᵢu′ᵢ⟩ₜ"] = ds["⟨uᵢuᵢ⟩ₜ"] - ds["ūᵢūᵢ"]

    # Normalize Unicode names in one place
    ds = normalize_unicode_names_in_dataset(ds, normalize_unicode)
    return ds

def get_shear_production_rates(ds, normalize_unicode=True):
    """Calculate shear production rates"""
    ds["SPR"] = - (ds["⟨u′ⱼu′ᵢ⟩ₜ"] * ds["∂ⱼūᵢ"]).sum("i")

    # Normalize Unicode names in one place
    ds = normalize_unicode_names_in_dataset(ds, normalize_unicode)
    return ds

def get_buoyancy_production_rates(ds, normalize_unicode=True):
    """Calculate buoyancy production rates"""
    ds["w̄b̄"]      = ds["ūᵢ"].sel(i=3) * ds["b̄"]
    ds["⟨w′b′⟩ₜ"] = ds["⟨wb⟩ₜ"] - ds["w̄b̄"]

    # Normalize Unicode names in one place
    ds = normalize_unicode_names_in_dataset(ds, normalize_unicode)
    return ds

def get_turbulent_kinetic_energy(ds, normalize_unicode=True):
    """Calculate turbulent kinetic energy"""
    if "⟨u′ᵢu′ᵢ⟩ₜ" in ds.variables: # If we have the diagonal of the tensor
        ds["⟨Ek′⟩ₜ"] = ds["⟨u′ᵢu′ᵢ⟩ₜ"].sum("i") / 2
    elif "⟨u′ⱼu′ᵢ⟩ₜ" in ds.variables: # If we have the full tensor
        ds["⟨Ek′⟩ₜ"] = (ds["⟨u′ⱼu′ᵢ⟩ₜ"].sel(i=1, j=1) +
                        ds["⟨u′ⱼu′ᵢ⟩ₜ"].sel(i=2, j=2) +
                        ds["⟨u′ⱼu′ᵢ⟩ₜ"].sel(i=3, j=3)) / 2
    else:
        raise(ValueError("No velocity variance variable found"))

    # Normalize Unicode names in one place
    ds = normalize_unicode_names_in_dataset(ds, normalize_unicode)
    return ds
#---
