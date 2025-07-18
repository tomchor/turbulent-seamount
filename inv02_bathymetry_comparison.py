import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
import numpy as np
import pynanigans as pn
import xarray as xr
from cycler import cycler
from matplotlib import pyplot as plt
from aux00_utils import merge_datasets
from aux01_physfuncs import fit_piecewise_powerlaw_spectrum
import xrft

plt.rcParams["figure.constrained_layout.use"] = True

#+++ Options
plot_results = False  # Set to False to skip plotting
save_results = True  # Set to False to skip saving results
#---

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

#+++ Fit piecewise power law to each spectrum using function from aux01_physfuncs
fit_results = {}
fitted_spectra = {}

for L_val in aaai.L.values:
    # Get spectrum and wavenumber for this L value
    spectrum = aaai.S_h.sel(L=L_val)
    k = spectrum.freq_r.values
    S = spectrum.values

    # Use the function from aux01_physfuncs to fit the spectrum
    result = fit_piecewise_powerlaw_spectrum(
        k, S,
        initial_k_transition=1/aaai.FWHM,  # Use FWHM as initial guess
        alpha_0=-2.0,
        debug=False  # Set to True for debugging individual fits
    )

    # Store results
    fit_results[L_val] = result
    fitted_spectra[L_val] = result["fitted_spectrum"]

    if result["fit_success"]:
        print(f"L={L_val}: α={result["alpha"]:.2f}, k_trans={result["k_transition"]:.4f} (fitted in log-log space)")
    else:
        print(f"L={L_val}: Fit failed")

#---

#+++ Save curve fitting results to dataset
if save_results:
    print("\nSaving curve fitting results...")

    # Create arrays for the fit parameters
    L_values = list(aaai.L.values)
    amps = []
    alphas = []
    k_transitions = []
    fit_successes = []

    for L_val in L_values:
        result = fit_results[L_val]
        amps.append(result["amp"] if result["fit_success"] else np.nan)
        alphas.append(result["alpha"] if result["fit_success"] else np.nan)
        k_transitions.append(result["k_transition"] if result["fit_success"] else np.nan)
        fit_successes.append(result["fit_success"])

        # Create dataset with fit results, adding back singleton dimensions
    # Extract parameter values from the original dataset
    Ro_h_val = float(aaai.Ro_h.values)
    Fr_h_val = float(aaai.Fr_h.values)
    dz_val = float(aaai.dz.values)
    closure_val = str(aaai.closure.values)

    fit_ds = xr.Dataset({
        "amplitude": (["Ro_h", "Fr_h", "dz", "closure", "L"],
                      np.array(amps).reshape(1, 1, 1, 1, len(amps))),
        "power_law_exponent": (["Ro_h", "Fr_h", "dz", "closure", "L"],
                               np.array(alphas).reshape(1, 1, 1, 1, len(alphas))),
        "transition_wavenumber": (["Ro_h", "Fr_h", "dz", "closure", "L"],
                                  np.array(k_transitions).reshape(1, 1, 1, 1, len(k_transitions))),
        "fit_success": (["Ro_h", "Fr_h", "dz", "closure", "L"],
                        np.array(fit_successes).reshape(1, 1, 1, 1, len(fit_successes))),
    }, coords={
        "Ro_h": [Ro_h_val],
        "Fr_h": [Fr_h_val],
        "dz": [dz_val],
        "closure": [closure_val],
        "L": L_values
    })

    # Add attributes
    fit_ds.attrs.update({
        "description": "Piecewise power law fit results for bathymetry spectra",
        "fit_function": "S(k) = amp * k^alpha for k < k_transition, amp * k_transition^alpha for k >= k_transition",
        "fitting_method": "log-log space using scipy.optimize.curve_fit",
        "FWHM": float(aaai.FWHM),
        "creation_date": str(np.datetime64("today"))
    })

    fit_ds["amplitude"].attrs = {
        "long_name": "Power law amplitude parameter",
        "units": "same as spectrum units"
    }
    fit_ds["power_law_exponent"].attrs = {
        "long_name": "Power law exponent (slope)",
        "units": "dimensionless"
    }
    fit_ds["transition_wavenumber"].attrs = {
        "long_name": "Transition wavenumber between power law and constant",
        "units": "rad/m"
    }
    fit_ds["fit_success"].attrs = {
        "long_name": "Whether curve fitting was successful",
        "units": "boolean"
    }

    # Add coordinate attributes
    fit_ds["Ro_h"].attrs = {
        "long_name": "Horizontal Rossby number",
        "units": "dimensionless"
    }
    fit_ds["Fr_h"].attrs = {
        "long_name": "Horizontal Froude number",
        "units": "dimensionless"
    }
    fit_ds["dz"].attrs = {
        "long_name": "Vertical grid spacing",
        "units": "m"
    }
    fit_ds["closure"].attrs = {
        "long_name": "Turbulence closure scheme",
        "units": "dimensionless"
    }
    fit_ds["L"].attrs = {
        "long_name": "Roughness length scale",
        "units": "m"
    }

    # Save to data_post directory
    import os
    os.makedirs("data_post", exist_ok=True)
    output_filename = f"data_post/bathymetry_powerlaw_fits_{simname_base}.nc"
    fit_ds.to_netcdf(output_filename)
    print(f"Saved curve fitting results to: {output_filename}")
#---

#+++ Plot results
if plot_results:
    print("\nGenerating plots...")

    fig, axes = plt.subplots(ncols=2, figsize=(12, 5))

    aaai.bottom_height.sel(x_caa=0, method="nearest").plot(hue="L", ax=axes[0])
    axes[0].legend_.texts[-1].set_text(f"Gaussian (FWHM={aaai.FWHM:.2f} m)")

    # Plot spectra and fitted curves
    spectra_plot = aaai.S_h.plot(hue="L", ax=axes[1], xscale="log", yscale="log")

    # Add fitted curves with matching colors but different line style
    for i, L_val in enumerate(aaai.L.values):
        if fit_results[L_val]["fit_success"]:
            color = axes[1].get_lines()[i].get_color()  # Match the spectrum color
            k = aaai.S_h.sel(L=L_val).freq_r.values
            axes[1].plot(k, fitted_spectra[L_val], color=color, linestyle="--", alpha=0.8, linewidth=2)

            # Add vertical line at transition point
            k_trans = fit_results[L_val]["k_transition"]
            axes[1].axvline(k_trans, color=color, linestyle=":", alpha=0.6, linewidth=1)

    axes[1].axvline(1/aaai.FWHM, color="black", linestyle="--", label="FWHM")

    # Create custom legend entries for the fit lines
    import matplotlib.lines as mlines
    fit_line = mlines.Line2D([], [], color="gray", linestyle="--", label="Piecewise fits (log-log)")
    trans_line = mlines.Line2D([], [], color="gray", linestyle=":", label="Transition points")

    # Get existing legend and add new entries
    handles, labels = axes[1].get_legend_handles_labels()
    handles.extend([fit_line, trans_line])
    labels.extend(["Piecewise fits (log-log)", "Transition points"])
    axes[1].legend(handles=handles, labels=labels)

    plt.show()
    print("Plots displayed successfully.")
else:
    print("\nSkipping plots (plot_results=False)")

# Print fit summary
print("\n=== Fit Results Summary (Log-log space fitting) ===")
for L_val in aaai.L.values:
    if fit_results[L_val]["fit_success"]:
        result = fit_results[L_val]
        label = "Gaussian" if L_val == aaai.L.values[-1] else f"L={L_val}"
        print(f"{label:>12}: α={result["alpha"]:>6.2f}, k_trans={result["k_transition"]:>8.4f}")

print(f"\nProcessing completed:")
print(f"  - Plotting: {"✓" if plot_results else "✗"}")
print(f"  - Saving results: {"✓" if save_results else "✗"}")
if save_results:
    # Get parameter values for filename (will be available from previous computation)
    Ro_h_val = float(aaai.Ro_h.values)
    Fr_h_val = float(aaai.Fr_h.values)
    dz_val = float(aaai.dz.values)
    closure_val = str(aaai.closure.values)
    print(f"  - Output file: data_post/bathymetry_powerlaw_fits_{simname_base}_Ro_h={Ro_h_val}_Fr_h={Fr_h_val}_dz={int(dz_val)}_closure={closure_val}.nc")
#---