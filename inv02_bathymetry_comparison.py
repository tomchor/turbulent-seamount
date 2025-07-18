import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
import numpy as np
import pynanigans as pn
import xarray as xr
from cycler import cycler
from matplotlib import pyplot as plt
from aux00_utils import merge_datasets
import xrft
from scipy.optimize import curve_fit

plt.rcParams["figure.constrained_layout.use"] = True

#+++ Define directory and simulation name
path = "simulations/data/"
simname_base = "seamount"

Rossby_numbers = cycler(Ro_h = [0.2])
Froude_numbers = cycler(Fr_h = [1.25])
L              = cycler(L = [0, 20, 40, 80, 160, 320])

resolutions    = cycler(dz = [2])
closures       = cycler(closure = ["DSM"])

paramspace = Rossby_numbers * Froude_numbers * L
configs    = resolutions * closures

runs = paramspace * configs
#---

#+++ Load aaai datasets
print("Loading aaai datasets...")
aaai = merge_datasets(runs, base_name=f"aaai.{simname_base}", dirpath=path, verbose=True, add_min_spacings=False).squeeze()
aaai = aaai[["bottom_height"]]
#---

#+++ Create Gaussian seamount with height H and width FWHM and merge with bottom height
r = np.sqrt(aaai.x_caa**2 + aaai.y_aca**2)
h_gaussian = aaai.H * np.exp(-np.log(2) * (r/aaai.FWHM)**2)

# Merge the datasets along L dimension
L_gaussian = 2 * aaai.L.values[-1]
aaai = aaai.reindex(L=list(aaai.L.values) + [L_gaussian])
aaai["bottom_height"].loc[dict(L=L_gaussian)] = h_gaussian.T
#---

#+++ Compute FFT and 2D power spectrumof bottom height
aaai["h_fft"] = xrft.fft(aaai.bottom_height, true_phase=True, true_amplitude=True, dim=("x_caa", "y_aca"))
aaai["h_fft_amp"] = np.abs(aaai.h_fft)

aaai["S_h_2d"] = (aaai.h_fft * np.conjugate(aaai.h_fft)).real
#---

#+++ Compute isotropic power spectrum of bottom height and Gaussian seamount
aaai["S_h"] = xrft.isotropic_power_spectrum(aaai.bottom_height,
                                            dim = ["x_caa", "y_aca"],
                                            nfactor = 4, # Number of radial bins relative to data size
                                            truncate = True, # Truncate at Nyquist frequency
                                            true_phase = True, # Preserve phase information
                                            true_amplitude = True # Proper amplitude scaling
                                            )

#---

#+++ Define and fit piecewise power law function
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

# Fit piecewise power law to each spectrum
fit_results = {}
fitted_spectra = {}

for L_val in aaai.L.values:
    # Get spectrum and wavenumber for this L value
    spectrum = aaai.S_h.sel(L=L_val)
    k = spectrum.freq_r.values
    S = spectrum.values

    # Remove zero or negative values for log fitting
    valid_mask = (S > 0) & (k > 0)
    k_fit = k[valid_mask]
    S_fit = S[valid_mask]

    # Convert to log space
    log_k = np.log(k_fit)
    log_S = np.log(S_fit)

    # Initial parameter guesses in log space
    # log_amp: log amplitude, alpha: power law exponent, log_k_transition: log transition wavenumber
    alpha_0 = -2
    log_amp_0 = np.log(S_fit[0]) - alpha_0 * log_k[0]  # Consistent with power law at first point
    log_k_trans_0 = np.log(1/aaai.FWHM)
    p0 = [log_amp_0, alpha_0, log_k_trans_0]

    # Bounds in log space: log_amp (any), alpha<0 (decay), log_k_transition (within data range)
    bounds = ([-np.inf, -10, log_k[0]],
              [np.inf, 0, log_k[-1]])

    # Fit the piecewise function in log-log space
    popt_log, pcov_log = curve_fit(piecewise_powerlaw_log, log_k, log_S, p0=p0, bounds=bounds, method="dogbox")

    # Convert back to linear space parameters
    log_amp, alpha, log_k_transition = popt_log
    amp = np.exp(log_amp)
    k_transition = np.exp(log_k_transition)

    # Store linear space parameters for compatibility
    popt = [amp, alpha, k_transition]

    # Store results
    fit_results[L_val] = {
        'amp': popt[0],
        'alpha': popt[1],
        'k_transition': popt[2],
        'fit_success': True
    }

    # Generate fitted spectrum for plotting on original k grid (all points including zeros)
    fitted_spectra[L_val] = piecewise_powerlaw(k, *popt)

    if False:
        # Plot original and fitted spectra for debugging
        plt.figure()
        plt.loglog(k_fit, S_fit, 'k-', label='Original (valid points)')
        plt.loglog(k, fitted_spectra[L_val], 'r--', label='Fitted (full grid)')
        plt.axvline(k_transition, color='g', linestyle=':', label='Transition')
        plt.legend()
        plt.xlabel('Wavenumber k')
        plt.ylabel('Spectrum S')
        plt.title(f'L = {L_val} (Log-log fit)')
        plt.show()

    print(f"L={L_val}: α={popt[1]:.2f}, k_trans={popt[2]:.4f} (fitted in log-log space)")

#---

#+++ Plot results
fig, axes = plt.subplots(ncols=2, figsize=(12, 5))

aaai.bottom_height.sel(x_caa=0, method="nearest").plot(hue="L", ax=axes[0])
axes[0].legend_.texts[-1].set_text(f"Gaussian (FWHM={aaai.FWHM:.2f} m)")

# Plot spectra and fitted curves
spectra_plot = aaai.S_h.plot(hue="L", ax=axes[1], xscale="log", yscale="log")

# Add fitted curves with matching colors but different line style
for i, L_val in enumerate(aaai.L.values):
    if fit_results[L_val]['fit_success']:
        color = axes[1].get_lines()[i].get_color()  # Match the spectrum color
        k = aaai.S_h.sel(L=L_val).freq_r.values
        axes[1].plot(k, fitted_spectra[L_val], color=color, linestyle='--', alpha=0.8, linewidth=2)

        # Add vertical line at transition point
        k_trans = fit_results[L_val]['k_transition']
        axes[1].axvline(k_trans, color=color, linestyle=':', alpha=0.6, linewidth=1)

axes[1].axvline(1/aaai.FWHM, color="black", linestyle="--", label="FWHM")

# Create custom legend entries for the fit lines
import matplotlib.lines as mlines
fit_line = mlines.Line2D([], [], color='gray', linestyle='--', label='Piecewise fits (log-log)')
trans_line = mlines.Line2D([], [], color='gray', linestyle=':', label='Transition points')

# Get existing legend and add new entries
handles, labels = axes[1].get_legend_handles_labels()
handles.extend([fit_line, trans_line])
labels.extend(['Piecewise fits (log-log)', 'Transition points'])
axes[1].legend(handles=handles, labels=labels)

# Print fit summary
print("\n=== Fit Results Summary (Log-log space fitting) ===")
for L_val in aaai.L.values:
    if fit_results[L_val]['fit_success']:
        result = fit_results[L_val]
        label = "Gaussian" if L_val == aaai.L.values[-1] else f"L={L_val}"
        print(f"{label:>12}: α={result['alpha']:>6.2f}, k_trans={result['k_transition']:>8.4f}")
#---