import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
from os.path import basename
import numpy as np
import xarray as xr
from cycler import cycler
import pynanigans as pn
from aux00_utils import open_simulation, condense, adjust_times, aggregate_parameters
from colorama import Fore, Back, Style
from dask.diagnostics import ProgressBar
xr.set_options(display_width=140, display_max_rows=30)

print("Starting energy transfer script")

#+++ Define directory and simulation name
if basename(__file__) != "00_run_postproc.py":
    path = "simulations/data/"
    simname_base = "seamount"

    slopes         = cycler(α = [0.05, 0.2])
    Rossby_numbers = cycler(Ro_h = [0.5])
    Froude_numbers = cycler(Fr_h = [0.2])

    resolutions    = cycler(res = [8, 4, 2])
    closures       = cycler(closure = ["AMD", "CSM", "DSM", "NON"])
    bcs            = cycler(bounded = [0])

    paramspace = slopes * Rossby_numbers * Froude_numbers
    configs    = resolutions * closures * bcs

    runs = paramspace * configs
#---

#+++ Options
indices = [1, 2, 3]
#---

for j, config in enumerate(runs):
    simname = f"{simname_base}_" + aggregate_parameters(config, sep="_", prefix="")

    #+++ Open datasets
    print(f"\nOpening {simname} xyz")
    xyz = open_simulation(path+f"xyz.{simname}.nc",
                                    use_advective_periods = True,
                                    topology = simname[:3],
                                    squeeze = True,
                                    load = False,
                                    get_grid = False,
                                    open_dataset_kwargs = dict(chunks="auto"),
                                    )
    print(f"Opening {simname} xyi")
    xyi = open_simulation(path+f"xyi.{simname}.nc",
                                    use_advective_periods = True,
                                    topology = simname[:3],
                                    squeeze = True,
                                    load = False,
                                    get_grid = False,
                                    open_dataset_kwargs = dict(chunks="auto"),
                                    )
#    print(f"Opening {simname} ttt")
#    ttt = open_simulation(path+f"ttt.{simname}.nc",
#                                    use_advective_periods = True,
#                                    topology = simname[:3],
#                                    squeeze = True,
#                                    load = False,
#                                    get_grid = False,
#                                    open_dataset_kwargs = dict(chunks="auto"),
#                                    )
    print(f"Opening {simname} tti")
    tti = open_simulation(path+f"tti.{simname}.nc",
                                    use_advective_periods = True,
                                    topology = simname[:3],
                                    squeeze = True,
                                    load = False,
                                    get_grid = False,
                                    open_dataset_kwargs = dict(chunks="auto"),
                                    )
    #---

    #+++ Get rid of slight misalignment in times
    xyz = adjust_times(xyz, round_times=True)
    xyi = adjust_times(xyi, round_times=True)
    #ttt = adjust_times(ttt, round_times=True)
    tti = adjust_times(tti, round_times=True)

    #ttt = ttt.assign_coords(xC=tti.xC.values, yC=tti.yC.values) # This is needed just as long as ttt is float32 and tti is float64
    #---

    #+++ Preliminary definitions and checks
    print("Doing prelim checks")
    Δt = xyz.time.diff("time").median()
    Δt_tol = Δt/100
    if np.all(xyi.time.diff("time") > Δt_tol):
        print(Fore.GREEN + f"Δt is consistent for {simname}", Style.RESET_ALL)
    else:
        print(f"Δt is inconsistent for {simname}")
        print(np.count_nonzero(xyz.time.diff("time") < Δt_tol), "extra time steps")

        tslice1 = slice(0.0, None, None)
        xyz = xyz.sel(time=tslice1)

        xyi = xyi.reindex(dict(time=np.arange(0, xyz.time[-1]+1e-5, Δt)), method="nearest", tolerance=Δt/Δt_tol)
        xyz = xyz.reindex(dict(time=np.arange(0, xyz.time[-1]+1e-5, Δt)), method="nearest", tolerance=Δt/Δt_tol)
    #---

    #+++ Trimming domain
    t_slice_inclusive = slice(tti.T_advective_spinup, np.inf) # For snapshots, we want to include t=T_advective_spinup
    t_slice_exclusive = slice(tti.T_advective_spinup + 0.01, np.inf) # For time-averaged outputs, we want to exclude t=T_advective_spinup
    x_slice = slice(xyz.xF[0], np.inf)
    y_slice = slice(None, np.inf)

    xyz = xyz.sel(time=t_slice_inclusive, xC=x_slice, xF=x_slice, yC=y_slice, yF=y_slice)
    xyi = xyi.sel(time=t_slice_inclusive, xC=x_slice, xF=x_slice, yC=y_slice, yF=y_slice)
    #ttt = ttt.sel(time=t_slice_exclusive, xC=x_slice, xF=x_slice, yC=y_slice, yF=y_slice)
    tti = tti.sel(time=t_slice_exclusive, xC=x_slice, xF=x_slice, yC=y_slice, yF=y_slice)
    #---

    #+++ Condense tensors
    def condense_velocities(ds):
        return condense(ds, ["u", "v", "w"], "uᵢ", dimname="i", indices=indices)

    def condense_velocity_gradient_tensor(ds):
        ds = condense(ds, ["∂u∂x", "∂v∂x", "∂w∂x"], "∂₁uᵢ", dimname="i", indices=indices)
        ds = condense(ds, ["∂u∂y", "∂v∂y", "∂w∂y"], "∂₂uᵢ", dimname="i", indices=indices)
        ds = condense(ds, ["∂u∂z", "∂v∂z", "∂w∂z"], "∂₃uᵢ", dimname="i", indices=indices)
        ds = condense(ds, ["∂₁uᵢ", "∂₂uᵢ", "∂₃uᵢ"], "∂ⱼuᵢ", dimname="j", indices=indices)
        return ds

    def condense_reynolds_stress_tensor(ds):
        ds["vu"] = ds.uv
        ds["wv"] = ds.vw
        ds["wu"] = ds.uw
        ds = condense(ds, ["uu",   "uv",   "uw"],   "u₁uᵢ", dimname="i", indices=indices)
        ds = condense(ds, ["vu",   "vv",   "vw"],   "u₂uᵢ", dimname="i", indices=indices)
        ds = condense(ds, ["wu",   "wv",   "ww"],   "u₃uᵢ", dimname="i", indices=indices)
        ds = condense(ds, ["u₁uᵢ", "u₂uᵢ", "u₃uᵢ"], "uⱼuᵢ", dimname="j", indices=indices)
        return ds

    #ttt = condense_velocities(ttt)
    #ttt = condense_velocity_gradient_tensor(ttt)
    #ttt = condense_reynolds_stress_tensor(ttt)
    tti = condense_velocities(tti)
    tti = condense_velocity_gradient_tensor(tti)
    tti = condense_reynolds_stress_tensor(tti)
    tti = condense(tti, ["dbdx", "dbdy", "dbdz"], "∂ⱼb", dimname="j", indices=indices)
    #---

    #+++ Time average
    # Here ū and ⟨u⟩ₜ are interchangeable
    tafields = tti.mean("time")
    tafields = tafields.rename({"uᵢ"     : "ūᵢ",
                                "∂ⱼuᵢ"   : "∂ⱼūᵢ",
                                "uⱼuᵢ"   : "⟨uⱼuᵢ⟩ₜ",
                                "b"      : "b̄",
                                "∂ⱼb"    : "∂ⱼb̄",
                                "wb"     : "⟨wb⟩ₜ",
                                "εₖ"     : "ε̄ₖ",
                                "εₚ"     : "ε̄ₚ",
                                "ν"      : "ν̄",
                                "κ"      : "κ̄",
                                "Ek"     : "⟨Ek⟩ₜ",
                                "PV"     : "q̄",
                                "∭⁵εₖdV" : "∭⁵ε̄ₖdV",
                                "∭⁵εₚdV" : "∭⁵ε̄ₚdV",
                                "∭⁵wbdV" : "⟨∭⁵wbdV⟩ₜ",
                                })
    tafields.attrs = tti.attrs
    #---

    #+++ Get turbulent Reynolds stress tensor
    tafields["ūⱼūᵢ"]      = tafields["ūᵢ"] * tafields["ūᵢ"].rename(i="j")
    tafields["⟨u′ⱼu′ᵢ⟩ₜ"] = tafields["⟨uⱼuᵢ⟩ₜ"] - tafields["ūⱼūᵢ"]
    #---

    #+++ Get shear production rates
    tafields["SPR"]  = - (tafields["⟨u′ⱼu′ᵢ⟩ₜ"] * tafields["∂ⱼūᵢ"]).sum("i")
    #---

    #+++ Get buoyancy production rates
    tafields["w̄b̄"]      = tafields["ūᵢ"].sel(i=3) * tafields["b̄"]
    tafields["⟨w′b′⟩ₜ"] = tafields["⟨wb⟩ₜ"] - tafields["w̄b̄"]
    #---

    #+++ Get TKE
    tafields["⟨Ek′⟩ₜ"] = (tafields["⟨u′ⱼu′ᵢ⟩ₜ"].sel(i=1, j=1) + tafields["⟨u′ⱼu′ᵢ⟩ₜ"].sel(i=2, j=2) + tafields["⟨u′ⱼu′ᵢ⟩ₜ"].sel(i=3, j=3)) / 2
    #---

    #+++ Volume-average/integrate results so far
    #tafields["ΔxΔyΔz"] = tafields["Δxᶜᶜᶜ"] * tafields["Δyᶜᶜᶜ"] * tafields["Δzᶜᶜᶜ"]
    #tafields["ΔxΔz"]   = tafields["Δxᶜᶜᶜ"] * tafields["Δzᶜᶜᶜ"]
    #def integrate(da, dV=tafields["ΔxΔyΔz"], dims=("x", "y", "z")):
    #    return (da*dV).pnsum(dims)

    #tafields["1"] = xr.ones_like(tafields["Δxᶜᶜᶜ"])
    #buffer = 5 # meters

    #distance_mask = tafields.altitude > buffer
    #for var in ["ε̄ₖ", "ε̄ₚ", "⟨∂ₜEk⟩ₜ", "⟨wb⟩ₜ", "⟨Ek⟩ₜ", "SPR", "w̄b̄", "⟨w′b′⟩ₜ", "⟨Ek′⟩ₜ", "κ̄ₑ", "1"]:
    #    int_all = f"∫∫∫⁰{var}dxdydz"
    #    int_buf = f"∫∫∫⁵{var}dxdydz"
    #    tafields[int_all] = integrate(tafields[var])
    #    tafields[int_buf] = integrate(tafields[var], dV=tafields.ΔxΔyΔz.where(distance_mask))
    #    tafields = condense(tafields, [int_all, int_buf], f"∫∫∫ᵇ{var}dxdydz", dimname="buffer", indices=[0, buffer])

    #for var in ["ε̄ₖ", "ε̄ₚ", "SPR", "⟨w′b′⟩ₜ", "⟨Ek′⟩ₜ", "1"]:
    #    int_all = f"∫∫⁰{var}dxdz"
    #    int_buf = f"∫∫⁵{var}dxdz"
    #    tafields[int_all] = integrate(tafields[var], dV=tafields.ΔxΔz, dims=("x", "z"))
    #    tafields[int_buf] = integrate(tafields[var], dV=tafields.ΔxΔz.where(distance_mask), dims=("x", "z"))
    #    tafields = condense(tafields, [int_all, int_buf], f"∫∫ᵇ{var}dxdz", dimname="buffer", indices=[0, buffer])

    #tafields["average_turbulence_mask"] = tafields["ε̄ₖ"] > 1e-10
    #for var in ["ε̄ₖ", "ε̄ₚ", "SPR", "⟨wb⟩ₜ", "1"]:
    #    int_turb = f"∫∫∫ᵋ{var}dxdydz"
    #    tafields[int_turb] = integrate(tafields[var], dV=tafields.ΔxΔyΔz.where(tafields.average_turbulence_mask))
    #---

    #+++ Get CSI mask and CSI-integral
    tafields["average_stratification_mask"] = tafields["∂ⱼb̄"].sel(j=3) > 0
    tafields["average_CSI_mask"] = ((tafields.q̄ * tafields.f_0) < 0) & (tafields["∂ⱼb̄"].sel(j=3) > 0)
    #---

    #+++ Drop unnecessary vars
    tafields = tafields.drop_vars(["ūⱼūᵢ", "⟨uⱼuᵢ⟩ₜ", "⟨u′ⱼu′ᵢ⟩ₜ",])
    tafields = tafields.drop_dims(("xF", "yF",))
    #---

    #+++ Save tafields
    outname = f"data_post/tafields_{simname}.nc"
    with ProgressBar(minimum=5, dt=5):
        print(f"Saving results to {outname}...")
        tafields.to_netcdf(outname)
        print("Done!\n")
    xyi.close(); xyz.close(); tti.close(); #ttt.close()
    #---

    #+++ Create bulk dataset
    bulk = xr.Dataset()
    bulk.attrs = tafields.attrs

    bulk["∭⁵ε̄ₖdV"]    = tafields["∭⁵ε̄ₖdV"]
    bulk["∭⁵ε̄ₚdV"]    = tafields["∭⁵ε̄ₚdV"]
    bulk["⟨∭⁵wbdV⟩ₜ"] = tafields["⟨∭⁵wbdV⟩ₜ"]

    bulk["⟨∬Ek′dxdy⟩ₜ"] = tafields["⟨Ek′⟩ₜ"].pnintegrate(("x", "y"))
    bulk["⟨∬Πdxdy⟩ₜ"]   = tafields["SPR"].sum("j").pnintegrate(("x", "y"))

    altitude = xyz.altitude.pnsel(z=tti.zC, method="nearest")
    bulk["⟨∬⁵Ek′dxdy⟩ₜ"] = tafields["⟨Ek′⟩ₜ"].where(altitude > 5, other=0).pnintegrate(("x", "y"))
    bulk["⟨∬⁵Πdxdy⟩ₜ"]   = tafields["SPR"].sum("j").where(altitude > 5, other=0).pnintegrate(("x", "y"))

    bulk["Slope_Bu"] = bulk.Slope_Bu
    bulk["Ro_h"] = bulk.Ro_h
    bulk["Fr_h"] = bulk.Fr_h
    bulk["α"]    = bulk.α

    bulk["Bu_h"] = bulk.Bu_h
    bulk["Γ"]    = bulk.Γ
    bulk["c_dz"] = bulk.c_dz

    bulk["f₀"]  = bulk.attrs["f₀"]
    bulk["N²∞"] = bulk.attrs["N²∞"]
    bulk["V∞"]  = bulk.attrs["V∞"]
    bulk["L"]   = bulk.L
    bulk["Δx_min"] = tafields["Δxᶜᶜᶜ"].where(tafields["Δxᶜᶜᶜ"] > 0).min().values
    bulk["Δy_min"] = tafields["Δyᶜᶜᶜ"].where(tafields["Δyᶜᶜᶜ"] > 0).min().values
    bulk["Δz_min"] = tafields["Δzᶜᶜᶜ"].where(tafields["Δzᶜᶜᶜ"] > 0).min().values

    bulk["RoFr"] = bulk.Ro_h * bulk.Fr_h
    bulk["V∞³÷L"] = bulk.attrs["V∞"]**3 / bulk.L
    bulk["V∞²N∞"] = bulk.attrs["V∞"]**2 * np.sqrt(bulk.N2_inf)
    bulk["N∞³L²"] = np.sqrt(bulk.N2_inf)**3 * bulk.L**2
    #---

    #+++ Final touches and save bulk
    outname = f"data_post/bulkstats_{simname}.nc"
    with ProgressBar(minimum=2, dt=5):
        print(f"Saving bulk results to {outname}...")
        bulk.to_netcdf(outname)
    #---
