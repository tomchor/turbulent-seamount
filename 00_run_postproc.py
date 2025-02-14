import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
import subprocess
import numpy as np
import xarray as xr
import pynanigans as pn
from cycler import cycler
from aux00_utils import check_simulation_completion
from aux01_physfuncs import get_topography_masks
from colorama import Fore, Back, Style
from dask.diagnostics import ProgressBar

print("Starting h00 script")

#+++ Define run options
simname_base = "tokara"

resolutions    = cycler(res = [8, 4, 2, 1])
resolutions    = cycler(res = [8, 4, 2])
slopes         = cycler(α = [0.05, 0.1, 0.2])
slopes         = cycler(α = [0.2,])
Rossby_numbers = cycler(Ro_h = [0.08, 0.2, 0.5, 1.25])
Froude_numbers = cycler(Fr_h = [0.08, 0.2, 0.5, 1.25])
closures       = cycler(closure = ["AMD"])
closures       = cycler(closure = ["CSM"])
bcs            = cycler(bounded = [0, 1,])

paramspace = slopes * Rossby_numbers * Froude_numbers
configs    = resolutions * closures * bcs

runs = resolutions * paramspace * closures * bcs
#---

for config in configs:
    print(config)
    continue
    run_options = aggregate_parameters(modifiers)
    simname = f"{simname_base}_" + aggregate_parameters(modifiers, sep="_", prefix="")
pause
