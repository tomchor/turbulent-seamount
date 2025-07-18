import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
import numpy as np
import pynanigans as pn
import xarray as xr
from cycler import cycler
from matplotlib import pyplot as plt
from aux00_utils import merge_datasets
import xrft

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

#+++ Compute characteristic length scale of bottom height and Gaussian seamount
k_cm = (aaai.S_h * aaai.S_h.freq_r).sum("freq_r") / aaai.S_h.sum("freq_r")
aaai["L_cm"] = 1/k_cm
#---

#+++ Plot results
fig, axes = plt.subplots(ncols=2, figsize=(12, 5))

aaai.bottom_height.sel(x_caa=0, method="nearest").plot(hue="L", ax=axes[0])

aaai.S_h.plot(hue="L", ax=axes[1], xscale="log", yscale="log")
axes[1].axvline(1/aaai.FWHM, color="black", linestyle="--", label="FWHM")
axes[1].axvline(2*np.pi/aaai.FWHM, color="black", linestyle="--", label="FWHM/2$\pi$")
