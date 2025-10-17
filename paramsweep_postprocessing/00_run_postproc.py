import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
sys.path.append("../postprocessing")
import numpy as np
from cycler import cycler
from src.aux00_utils import check_simulation_completion, aggregate_parameters
from colorama import Fore, Back, Style

print("Starting h00 script")

#+++ Define run options
path = "../simulations/data/"
simname_base = "seamount"

Rossby_numbers = cycler(Ro_b = [0.05, 0.1, 0.2, 0.5])
Froude_numbers = cycler(Fr_b = [0.05, 0.08, 0.3, 1, 2])
L              = cycler(L = [0, 0.8])

resolutions    = cycler(dz = [2])
T_adv_spinups  = cycler(T_adv_spinup = [12])

paramspace = Rossby_numbers * Froude_numbers * L
configs    = resolutions  * T_adv_spinups

runs = paramspace * configs
#---

for config in configs:
    print(config)
    config_suffix = aggregate_parameters(config, sep="_", prefix="")
    simnames = [ simname_base + "_" + aggregate_parameters(params, sep="_", prefix="") + "_" + config_suffix for params in paramspace ]
    print(simnames)
    check_simulation_completion(simnames, slice_name="xyzi", path="../simulations/data/", verbose=False)
    print()

print(Back.LIGHTWHITE_EX + Fore.BLUE + "\nStarting 01 post-processing of results using `configs`", Style.RESET_ALL, configs)
exec(open("01_create_aaaa.py").read())
