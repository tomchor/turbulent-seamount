import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
from cycler import cycler
from src.aux00_utils import check_simulation_completion, aggregate_parameters
from colorama import Fore, Back, Style

print("Starting h00 script")

#+++ Define run options
simdata_path = "../simulations/data/"
simname_base = "balanus"

Rossby_numbers = cycler(Ro_b = [0.1])
Froude_numbers = cycler(Fr_b = [1])
L              = cycler(L = [0, 0.05, 0.1, 0.2, 0.4, 0.8])
FWHM           = cycler(FWHM = [500, 500, 500, 500, 500, 500])

resolutions    = cycler(dz = [4, 2, 1])

paramspace = Rossby_numbers * Froude_numbers * (L + FWHM)
configs    = resolutions

runs = paramspace * configs
#---

for config in configs:
    print(config)
    config_suffix = aggregate_parameters(config, sep="_", prefix="")
    simnames = [ simname_base + "_" + aggregate_parameters(params, sep="_", prefix="") + "_" + config_suffix for params in paramspace ]
    print(simnames)
    check_simulation_completion(simnames, slice_name="xyzi", path=simdata_path, verbose=False)
    print()

print(Back.LIGHTWHITE_EX + Fore.BLUE + "\nStarting 01 post-processing of results using `configs`", Style.RESET_ALL, configs)
exec(open("01_create_aaaa.py").read())
exec(open("02_create_xyza.py").read())
exec(open("03_create_xyzd.py").read())
exec(open("04_create_aaad.py").read())
