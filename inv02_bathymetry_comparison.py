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

aaai["h_fft"] = xrft.fft(aaai.bottom_height, true_phase=True, true_amplitude=True, dim=("x_caa", "y_aca"))
aaai["h_fft_amp"] = np.abs(aaai.h_fft)

aaai["S_h"] = (aaai.h_fft * np.conjugate(aaai.h_fft)).real

Sh_pos = aaai.S_h.where(aaai.freq_y_aca>0, drop=True).sel(freq_x_caa=0, freq_y_aca=slice(0, None))
k_cm = (Sh_pos * Sh_pos.freq_y_aca).sum("freq_y_aca") / Sh_pos.sum("freq_y_aca")
aaai["L_cm"] = 1/k_cm

# Create Gaussian seamount with height H and width FWHM
x = aaai.x_caa
y = aaai.y_aca
X, Y = np.meshgrid(x, y, indexing='ij')
R = np.sqrt(X**2 + Y**2)

aaai["h_gaussian"] = aaai.H * np.exp(-4*np.log(2) * (R/aaai.FWHM)**2)
