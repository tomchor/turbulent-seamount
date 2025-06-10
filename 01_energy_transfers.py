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
π = 2*np.pi

print("Starting energy transfer script")

#+++ Define directory and simulation name
if basename(__file__) != "00_run_postproc.py":
    path = "simulations/data/"
    simname_base = "seamount"

    Rossby_numbers = cycler(Ro_h = [0.2, 1.25])
    Froude_numbers = cycler(Fr_h = [0.2, 1.25])
    L              = cycler(L = [0, 300])

    resolutions    = cycler(dz = [4,])
    closures       = cycler(closure = ["AMD", "AMC", "CSM", "DSM", "NON"])
    closures       = cycler(closure = ["AMD", "CSM", "DSM"])

    paramspace = Rossby_numbers * Froude_numbers * L
    configs    = resolutions * closures

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
    print(f"Opening {simname} ttt")
    ttt = open_simulation(path+f"ttt.{simname}.nc",
                                    use_advective_periods = True,
                                    topology = simname[:3],
                                    squeeze = False,
                                    load = False,
                                    get_grid = False,
                                    open_dataset_kwargs = dict(chunks="auto"),
                                    )
    print(f"Opening {simname} tti")
    tti = open_simulation(path+f"tti.{simname}.nc",
                                    use_advective_periods = True,
                                    topology = simname[:3],
                                    squeeze = False,
                                    load = False,
                                    get_grid = False,
                                    open_dataset_kwargs = dict(chunks="auto"),
                                    )
    #---

    #+++ Get rid of slight misalignment in times
    xyz = adjust_times(xyz, round_times=True)
    xyi = adjust_times(xyi, round_times=True)
    ttt = adjust_times(ttt, round_times=True)
    tti = adjust_times(tti, round_times=True)

    ttt = ttt.assign_coords(x_caa=tti.x_caa.values, y_aca=tti.y_aca.values) # This is needed just as long as ttt is float32 and tti is float64
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
    x_slice = slice(xyz.x_faa[0], np.inf)
    y_slice = slice(None, np.inf)

    xyz = xyz.sel(time=t_slice_inclusive, x_caa=x_slice, x_faa=x_slice, y_aca=y_slice, y_afa=y_slice)
    xyi = xyi.sel(time=t_slice_inclusive, x_caa=x_slice, x_faa=x_slice, y_aca=y_slice, y_afa=y_slice)
    ttt = ttt.sel(time=t_slice_exclusive, x_caa=x_slice, x_faa=x_slice, y_aca=y_slice, y_afa=y_slice)
    tti = tti.sel(time=t_slice_exclusive, x_caa=x_slice, x_faa=x_slice, y_aca=y_slice, y_afa=y_slice)
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

    ttt = condense_velocities(ttt)
    #ttt = condense_velocity_gradient_tensor(ttt)
    ttt = condense_reynolds_stress_tensor(ttt)
    tti = condense_velocities(tti)
    tti = condense_velocity_gradient_tensor(tti)
    tti = condense_reynolds_stress_tensor(tti)
    #tti = condense(tti, ["dbdx", "dbdy", "dbdz"], "∂ⱼb", dimname="j", indices=indices)
    #---

    #+++ Time average
    # Here ū and ⟨u⟩ₜ are interchangeable
    xyia = tti.mean("time")
    xyia = xyia.rename({#"uᵢ"      : "ūᵢ",
                        "∂ⱼuᵢ"    : "∂ⱼūᵢ",
                        #"uⱼuᵢ"    : "⟨uⱼuᵢ⟩ₜ",
                        #"b"       : "b̄",
                        #"∂ⱼb"     : "∂ⱼb̄",
                        "wb"      : "⟨wb⟩ₜ",
                        "εₖ"      : "ε̄ₖ",
                        "εₚ"      : "ε̄ₚ",
                        "ν"       : "ν̄",
                        "κ"       : "κ̄",
                        "Ek"      : "⟨Ek⟩ₜ",
                        "PV"      : "q̄",
                        "∭⁵εₖdV"  : "∭⁵ε̄ₖdV",
                        "∭⁵εₚdV"  : "∭⁵ε̄ₚdV",
                        "∭⁵wbdV"  : "⟨∭⁵wbdV⟩ₜ",
                        "∭¹⁰εₖdV" : "∭¹⁰ε̄ₖdV",
                        "∭¹⁰εₚdV" : "∭¹⁰ε̄ₚdV",
                        "∭¹⁰wbdV" : "⟨∭¹⁰wbdV⟩ₜ",
                        "∭²⁰εₖdV" : "∭²⁰ε̄ₖdV",
                        "∭²⁰εₚdV" : "∭²⁰ε̄ₚdV",
                        "∭²⁰wbdV" : "⟨∭²⁰wbdV⟩ₜ",
                        })
    xyia = xyia.drop(["uᵢ", "uⱼuᵢ"])
    xyia.attrs = tti.attrs

    xyza = ttt.mean("time").rename({"uᵢ"      : "ūᵢ",
                                    "uⱼuᵢ"    : "⟨uⱼuᵢ⟩ₜ",
                                    "b"       : "b̄",
                                    "εₖ"      : "ε̄ₖ",
                                    "εₚ"      : "ε̄ₚ",
                                    "κ"       : "κ̄",
                                    })
    xyza.attrs = tti.attrs
    #---

    #+++ Get turbulent Reynolds stress tensor
    xyza["ūⱼūᵢ"]      = xyza["ūᵢ"] * xyza["ūᵢ"].rename(i="j")
    xyza["⟨u′ⱼu′ᵢ⟩ₜ"] = xyza["⟨uⱼuᵢ⟩ₜ"] - xyza["ūⱼūᵢ"]
    #---

    #+++ Get shear production rates
    xyia["SPR"] = - (xyza["⟨u′ⱼu′ᵢ⟩ₜ"] * xyia["∂ⱼūᵢ"]).sum("i")
    #---

    #+++ Get buoyancy production rates
    xyia["w̄b̄"]      = xyza["ūᵢ"].sel(i=3) * xyza["b̄"]
    xyia["⟨w′b′⟩ₜ"] = xyia["⟨wb⟩ₜ"] - xyia["w̄b̄"]
    #---

    #+++ Get TKE
    xyza["⟨Ek′⟩ₜ"] = (xyza["⟨u′ⱼu′ᵢ⟩ₜ"].sel(i=1, j=1) + xyza["⟨u′ⱼu′ᵢ⟩ₜ"].sel(i=2, j=2) + xyza["⟨u′ⱼu′ᵢ⟩ₜ"].sel(i=3, j=3)) / 2
    #---

    #+++ Volume-average/integrate results so far
    xyza["ΔxΔyΔz"] = xyza["Δx_caa"] * xyza["Δy_aca"] * xyza["Δz_aac"]
    xyia["ΔxΔy"]   = xyia["Δx_caa"] * xyia["Δy_aca"]
    def integrate(da, dV = None, dims=("x", "y", "z")):
        if dV is None:
            if dims == ("x", "y", "z"):
                dV = xyia["ΔxΔyΔz"]
            elif dims == ("x", "y"):
                dV =  xyia["ΔxΔy"]
        return (da * dV).pnsum(dims)

    tafields["1"] = xr.ones_like(tafields["Δx_caa"])
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
    #    tafields[int_all] = integrate(tafields[var], dV=tafields.ΔxΔy, dims=("x", "z"))
    #    tafields[int_buf] = integrate(tafields[var], dV=tafields.ΔxΔy.where(distance_mask), dims=("x", "z"))
    #    tafields = condense(tafields, [int_all, int_buf], f"∫∫ᵇ{var}dxdz", dimname="buffer", indices=[0, buffer])

    tafields["average_turbulence_mask"] = tafields["ε̄ₖ"] > 1e-8
    for var in ["ε̄ₖ", "ε̄ₚ", "SPR", "⟨w′b′⟩ₜ", "1"]:
        int_turb = f"∬ᵋ{var}dxdy"
        tafields[int_turb] = integrate(tafields[var], dV=tafields.ΔxΔy.where(tafields.average_turbulence_mask), dims=("x", "y"))
    #---

    #+++ Get CSI mask and CSI-integral
    #tafields["average_stratification_mask"] = tafields["∂ⱼb̄"].sel(j=3) > 0
    #tafields["average_CSI_mask"] = ((tafields.q̄ * tafields.f_0) < 0) & (tafields["∂ⱼb̄"].sel(j=3) > 0)
    #---

    #+++ Drop unnecessary vars
    tafields = tafields.drop_vars(["ūⱼūᵢ", "⟨uⱼuᵢ⟩ₜ", "⟨u′ⱼu′ᵢ⟩ₜ",])
    tafields = tafields.drop_dims(("x_faa", "y_afa",))
    #---

    #+++ Save xyia
    outname = f"data_post/tafields_{simname}.nc"
    with ProgressBar(minimum=5, dt=5):
        print(f"Saving results to {outname}...")
        xyia.to_netcdf(outname)
        print("Done!\n")
    xyi.close(); xyz.close(); tti.close()
    #---

    #+++ Save xyza
    outname = f"data_post/xyza_{simname}.nc"
    with ProgressBar(minimum=5, dt=5):
        print(f"Saving results to {outname}...")
        xyza.to_netcdf(outname)
        print("Done!\n")
    xyza.close(); ttt.close()
    #---

    #+++ Create bulk dataset
    bulk = xr.Dataset()
    bulk.attrs = xyia.attrs

    bulk["∭⁵ε̄ₖdV"]     = xyia["∭⁵ε̄ₖdV"]
    bulk["∭⁵ε̄ₚdV"]     = xyia["∭⁵ε̄ₚdV"]
    bulk["⟨∭⁵wbdV⟩ₜ"]  = xyia["⟨∭⁵wbdV⟩ₜ"]
    bulk["∭¹⁰ε̄ₖdV"]    = xyia["∭¹⁰ε̄ₖdV"]
    bulk["∭¹⁰ε̄ₚdV"]    = xyia["∭¹⁰ε̄ₚdV"]
    bulk["⟨∭¹⁰wbdV⟩ₜ"] = xyia["⟨∭¹⁰wbdV⟩ₜ"]
    bulk["∭²⁰ε̄ₖdV"]    = xyia["∭²⁰ε̄ₖdV"]
    bulk["∭²⁰ε̄ₚdV"]    = xyia["∭²⁰ε̄ₚdV"]
    bulk["⟨∭²⁰wbdV⟩ₜ"] = xyia["⟨∭²⁰wbdV⟩ₜ"]
    pause

    bulk["⟨∬w′b′dxdy⟩ₜ"] = integrate(xyia["⟨w′b′⟩ₜ"], dims = ("x", "y"))
    bulk["⟨∬Ek′dxdy⟩ₜ"]  = integrate(xyia["⟨Ek′⟩ₜ"], dims = ("x", "y"))
    bulk["⟨∬SPRdxdy⟩ₜ"]  = integrate(xyia["SPR"], dims = ("x", "y"))
    bulk["⟨∬Πdxdy⟩ₜ"]    = bulk["⟨∬SPRdxdy⟩ₜ"].sum("j")

    altitude = xyz.altitude.pnsel(z=tti.z_aac, method="nearest")
    bulk["⟨∬⁵w′b′dxdy⟩ₜ"] = integrate(xyia["⟨w′b′⟩ₜ"].where(altitude > 5, other=0), dims = ("x", "y"))
    bulk["⟨∬⁵Ek′dxdy⟩ₜ"]  = integrate(xyia["⟨Ek′⟩ₜ"].where(altitude > 5, other=0), dims = ("x", "y"))
    bulk["⟨∬⁵SPRdxdy⟩ₜ"]  = integrate(xyia["SPR"].where(altitude > 5, other=0), dims = ("x", "y"))
    bulk["⟨∬⁵Πdxdy⟩ₜ"]    = bulk["⟨∬⁵SPRdxdy⟩ₜ"].sum("j")

    bulk["∬ᵋε̄ₖdxdy"] = xyia["∬ᵋε̄ₖdxdy"]
    bulk["⟨ε̄ₖ⟩ᵋ"]    = bulk["∬ᵋε̄ₖdxdy"] / xyia["∬ᵋ1dxdy"]
    bulk["Loᵋ"]      = 2*π * np.sqrt(bulk["⟨ε̄ₖ⟩ᵋ"] / bulk.N2_inf**(3/2))

    bulk["Δz̃"] = bulk.Δz_min / bulk["Loᵋ"]

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
    bulk["Δx_min"] = xyia["Δx_caa"].where(xyia["Δx_caa"] > 0).min().values
    bulk["Δy_min"] = xyia["Δy_aca"].where(xyia["Δy_aca"] > 0).min().values
    bulk["Δz_min"] = xyia["Δz_aac"].where(xyia["Δz_aac"] > 0).min().values

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
