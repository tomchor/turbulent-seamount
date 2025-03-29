import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
import subprocess
import numpy as np
import xarray as xr
import pynanigans as pn
from cycler import cycler
from aux00_utils import check_simulation_completion, aggregate_parameters
from aux01_physfuncs import get_topography_masks
from colorama import Fore, Back, Style
from dask.diagnostics import ProgressBar

print("Starting h00 script")

#+++ Define run options
path = "simulations/data/"
simname_base = "seamount"

slopes         = cycler(α = [0.05, 0.2])
Rossby_numbers = cycler(Ro_h = [0.5])
Froude_numbers = cycler(Fr_h = [0.2])

resolutions    = cycler(dz = [8, 4, 2])
closures       = cycler(closure = ["AMD", "CSM", "DSM", "NON", "AMC"])
closures       = cycler(closure = ["AMD", "DSM"])

paramspace = slopes * Rossby_numbers * Froude_numbers
configs    = resolutions * closures

runs = paramspace * configs
#---

for config in configs:
    print(config)
    config_suffix = aggregate_parameters(config, sep="_", prefix="")
    simnames = [ simname_base + "_" + aggregate_parameters(params, sep="_", prefix="") + "_" + config_suffix for params in paramspace ]
    print(simnames)
    check_simulation_completion(simnames, slice_name="tti", path="simulations/data/")
    print()

print(Back.LIGHTWHITE_EX + Fore.BLUE + "\nStarting 01 post-processing of results using `configs`", Style.RESET_ALL, configs)
exec(open("01_energy_transfers.py").read())
