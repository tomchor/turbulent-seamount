import xarray as xr
import pynanigans as pn
import numpy as np
import unicodedata

#+++ Unicode normalization functions
def normalize_unicode_name(name, form="NFD"):
    """
    Normalize a Unicode variable name using unicodedata.normalize

    Parameters
    ----------
    name : str
        The variable name to normalize
    form : str, optional; default "NFD"
        The form of Unicode normalization to apply.
        See https://docs.python.org/3/library/unicodedata.html#unicodedata.normalize
        and https://en.wikipedia.org/wiki/Unicode_equivalence#Normal_forms
        for more information.

    Returns
    -------
    str
        The normalized variable name
    """
    return unicodedata.normalize(form, name)

def normalize_unicode_names_in_dataset(ds, normalize_unicode=True, form="NFD"):
    """
    Normalize all Unicode variable names in a dataset using unicodedata.normalize

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset containing variables to normalize
    normalize_unicode : bool, optional
        Whether to apply Unicode normalization. Default True.

    Returns
    -------
    xarray.Dataset
        The dataset with normalized variable names
    """
    if not normalize_unicode:
        return ds

    # Get all variable names that contain Unicode characters
    unicode_vars = []
    for var_name in ds.variables:
        if any(ord(char) > 127 for char in var_name):
            unicode_vars.append(var_name)

    # Create rename dictionary for Unicode variables
    rename_dict = {}
    for var_name in unicode_vars:
        normalized_name = normalize_unicode_name(var_name, form=form)
        if normalized_name != var_name:
            rename_dict[var_name] = normalized_name

    # Apply renaming if there are any Unicode variables to normalize
    if rename_dict:
        ds = ds.rename(rename_dict)

    return ds
#---

#+++ Open simulation following the standard way
def open_simulation(fname,
                    use_inertial_periods=False,
                    use_cycle_periods=False,
                    use_advective_periods=False,
                    use_strouhal_periods=False,
                    open_dataset_kwargs=dict(),
                    regularize_ds_kwargs=dict(),
                    load=False,
                    squeeze=True,
                    unique=True,
                    verbose=False,
                    get_grid = True,
                    topology="PPN", **kwargs):

    #+++ Open dataset and create grid before squeezing
    if verbose: print(f"\nOpening {fname}... ", end="")
    if load:
        ds = xr.load_dataset(fname, decode_times=False, **open_dataset_kwargs)
    else:
        ds = xr.open_dataset(fname, decode_times=False, **open_dataset_kwargs)
    if verbose: print("Done")
    #---

    #+++ Get grid
    if get_grid: grid_ds = pn.get_grid(ds, topology=topology, **kwargs)
    #---

    #+++ Squeeze?
    if squeeze: ds = ds.squeeze()
    #---

    #+++ Normalize units and regularize
    if use_inertial_periods:
        ds = pn.normalize_time_by(ds, seconds=ds.T_inertial, new_units="Inertial period")
    elif use_advective_periods:
        ds = pn.normalize_time_by(ds, seconds=ds.T_advective, new_units="Cycle period")
    elif use_cycle_periods:
        ds = pn.normalize_time_by(ds, seconds=ds.T_cycle, new_units="Cycle period")
    elif use_strouhal_periods:
        ds = pn.normalize_time_by(ds, seconds=ds.T_strouhal, new_units="Strouhal period")
    #---

    #+++ Returning only unique times:
    if unique:
        import numpy as np
        _, index = np.unique(ds['time'], return_index=True)
        if verbose and (len(index)!=len(ds.time)): print("Cleaning non-unique indices")
        ds = ds.isel(time=index)
    #---

    #+++ Return
    if get_grid:
        return grid_ds, ds
    else:
        return ds
    #---
#---

#+++ Condense variables into one (in datasets)
def condense(ds, vlist, varname, dimname="α", indices=None):
    """
    Condense variables in `vlist` into one variable named `varname`.
    In the process, individual variables in `vlist` are removed from `ds`.
    """
    if indices is None:
        indices = range(1, len(vlist)+1)

    ds[varname] = ds[vlist].to_array(dim=dimname).assign_coords({dimname : list(indices)})
    ds = ds.drop(vlist)
    return ds
#---

#+++ Tensor condensation functions
def condense_velocities(ds, indices=[1, 2, 3]):
    """Condense velocity components into tensor form"""
    return condense(ds, ["u", "v", "w"], "uᵢ", dimname="i", indices=indices)

def condense_velocity_gradient_tensor(ds, indices=[1, 2, 3]):
    """Condense velocity gradient components into tensor form"""
    ds = condense(ds, ["∂u∂x", "∂v∂x", "∂w∂x"], "∂₁uᵢ", dimname="i", indices=indices)
    ds = condense(ds, ["∂u∂y", "∂v∂y", "∂w∂y"], "∂₂uᵢ", dimname="i", indices=indices)
    ds = condense(ds, ["∂u∂z", "∂v∂z", "∂w∂z"], "∂₃uᵢ", dimname="i", indices=indices)
    ds = condense(ds, ["∂₁uᵢ", "∂₂uᵢ", "∂₃uᵢ"], "∂ⱼuᵢ", dimname="j", indices=indices)
    return ds

def condense_reynolds_stress_tensor(ds, indices=[1, 2, 3]):
    """Condense Reynolds stress components into tensor form"""
    ds["vu"] = ds.uv
    ds["wv"] = ds.vw
    ds["wu"] = ds.uw
    ds = condense(ds, ["uu",   "uv",   "uw"],   "u₁uᵢ", dimname="i", indices=indices)
    ds = condense(ds, ["vu",   "vv",   "vw"],   "u₂uᵢ", dimname="i", indices=indices)
    ds = condense(ds, ["wu",   "wv",   "ww"],   "u₃uᵢ", dimname="i", indices=indices)
    ds = condense(ds, ["u₁uᵢ", "u₂uᵢ", "u₃uᵢ"], "uⱼuᵢ", dimname="j", indices=indices)
    return ds
#---

#+++ Merge datasets into one
def merge_datasets(
        runs,
        base_name="seamount",
        dirpath="data_post",
        add_min_spacings=False,
        add_simulation_info=True,
        verbose=False,
        drop_vars=None,
        open_dataset_kwargs={},
        combine_by_coords_kwargs={"combine_attrs": "drop_conflicts"},
        adjust_times_before_merge=False):
    """
    Merge multiple datasets into one.

    Parameters
    ----------
    runs : cycler.Cycler
        Parameter space to iterate over
    base_name : str, optional
        Base name for simulation files. Default "seamount"
    dirpath : str, optional
        Directory path containing the datasets. Default "data_post"
    add_min_spacings : bool, optional
        Whether to add minimum spacing variables. Default False
    add_simulation_info : bool, optional
        Whether to add simulation information variables. Default True
    verbose : bool, optional
        Whether to print verbose output. Default False
    drop_vars : list of str, optional
        List of variable names to drop from each dataset before merging. Default None
    open_dataset_kwargs : dict, optional
        Additional keyword arguments to pass to xr.open_dataset. Default {}
    combine_by_coords_kwargs : dict, optional
        Additional keyword arguments to pass to xr.combine_by_coords.
        Default {"combine_attrs": "drop_conflicts"}
    adjust_times_before_merge : bool, optional
        Whether to call adjust_times() on each dataset before merging. Default False

    Returns
    -------
    xarray.Dataset
        Merged dataset
    """
    simnames_filtered = list(map(lambda run: form_run_names(base_name, run, sep="_", prefix=""), runs))
    dslist = []


    for sim_number, simname in enumerate(simnames_filtered):
        #+++ Open volume-integrated output
        fname = f"{simname}.nc"
        if verbose: print(f"\nOpening {fname}")
        ds = xr.open_dataset(f"{dirpath}/{fname}", **open_dataset_kwargs)
        #---

        #+++ Drop specified variables if requested
        if drop_vars is not None:
            # Only drop variables that exist in the dataset
            vars_to_drop = [var for var in drop_vars if var in ds.variables]
            if vars_to_drop:
                if verbose: print(f"Dropping variables: {vars_to_drop}")
                ds = ds.drop_vars(vars_to_drop)
        #---

        #+++ Create auxiliary variables and organize them into a Dataset
        if add_min_spacings: # Calculate resolutions before they get thrown out
            if "Δx_min" not in ds.keys(): ds["Δx_min"] = ds["Δx_caa"].where(ds["Δx_caa"] > 0).min().values
            if "Δy_min" not in ds.keys(): ds["Δy_min"] = ds["Δy_aca"].where(ds["Δy_aca"] > 0).min().values
            if "Δz_min" not in ds.keys(): ds["Δz_min"] = ds["Δz_aac"].where(ds["Δz_aac"] > 0).min().values

        if add_simulation_info:
            ds["simulation"] = simname
            ds["sim_number"] = sim_number

        for dim in runs.keys: # Let's make each variable in `runs` an xarray `coord`
            if dim not in ds.variables: # if `dim` is not a variable, make it one
                ds = ds.expand_dims((dim,))
            ds = ds.assign_coords({dim : [ds.attrs[dim]]})

        #+++ Optionally adjust times before merging
        if adjust_times_before_merge:
            ds = adjust_times(ds)
        #---

        dslist.append(ds)
        #---

    return xr.combine_by_coords(dslist, **combine_by_coords_kwargs)
#---

#+++ Time adjustment
def adjust_times(ds, round_times=True, decimals=4):
    import numpy as np
    from statistics import mode
    if len(ds.time) > 1:
        Δt = np.round(mode(ds.time.diff("time").values), decimals=5)
        ds = ds.sel(time=np.arange(ds.time[0], ds.time[-1]+Δt/2, Δt), method="nearest")

        if round_times:
            ds = ds.assign_coords(time = list( map(lambda x: np.round(x, decimals=decimals), ds.time.values) ))
    return ds
#---

#+++ Check if all simulations are complete
def check_simulation_completion(simnames, slice_name="xyza", path="./simulations/data/", verbose=True):
    from colorama import Fore, Back, Style
    times = []
    for simname in simnames:
        with open_simulation(path+f"{slice_name}.{simname}.nc", use_advective_periods = True, get_grid = False, verbose=verbose, squeeze=False) as ds:
            ds = adjust_times(ds, round_times=True)
            times.append(ds.time.values)
            print(simname, ds.time.values)
    message = Fore.GREEN + "All times equal" + Style.RESET_ALL
    for time in times[1:]:
        if (len(time)!=len(times[0])) or  (time != times[0]).any():
            message = Fore.RED + "Not all times are equal!" + Style.RESET_ALL
    print(message)
    return
#---

#+++ Aggregate parameters into strings
def aggregate_parameters(parameters, sep=" ", prefix="--", use_equals=False):
    """
    Aggregate parameters into a string representation.

    Parameters
    ----------
    parameters : dict
        Dictionary of parameters to aggregate
    sep : str, optional
        Separator between parameters. Default " "
    prefix : str, optional
        Prefix for each parameter. Default "--"
    use_equals : bool, optional
        Whether to use equals signs in parameter formatting.
        If None, will use equals signs when prefix is "--" (command line style),
        but not when prefix is "" (filename style). Default None.

    Returns
    -------
    str
        Aggregated parameter string
    """
    if use_equals is None:
        # Default behavior: use equals for command line (prefix="--"), not for filenames (prefix="")
        use_equals = (prefix == "--")

    if use_equals:
        written_out_list = [ f"{prefix}{key}={val}" for key, val in parameters.items() ]
    else:
        written_out_list = [ f"{prefix}{key}{val}" for key, val in parameters.items() ]
    return sep.join(written_out_list)

def form_run_names(superprefix, *args, **kwargs):
    return f"{superprefix}_" + aggregate_parameters(*args, **kwargs)
#---

#+++ Define collect_datasets() function
def collect_datasets(simnames_filtered, slice_name="xyii", path="./simulations/data/", verbose=False):
    dslist = []
    for sim_number, simname in enumerate(simnames_filtered):
        #+++ Open datasets
        #+++ Deal with time-averaged output
        if slice_name == "tafields":
            fname = f"tafields_{simname}.nc"
            print(f"\nOpening {fname}")
            ds = xr.open_dataset(f"data_post/{fname}", chunks=dict(time="auto", L="auto"))
        #---

        #+++ Deal with volume-integrated output
        elif slice_name == "turbstats":
            fname = f"turbstats_{simname}.nc"
            print(f"\nOpening {fname}")
            ds = xr.open_dataset(f"data_post/{fname}", chunks=dict(time="auto", L="auto"))
        #---

        #+++ Deal with snapshots
        else:
            fname = f"{slice_name}.{simname}.nc"
            print(f"\nOpening {fname}")
            ds = open_simulation(path + fname,
                                 use_advective_periods=True,
                                 topology=simname[:3],
                                 squeeze=True,
                                 load=False,
                                 get_grid = False,
                                 open_dataset_kwargs=dict(chunks=dict(time=1)),
                                 )

            if slice_name == "xyii":
                ds = ds.drop_vars(["z_aac", "z_aaf"])
            elif slice_name == "xiza":
                ds = ds.drop_vars(["y_aca", "y_afa"])
            elif slice_name == "iyz":
                ds = ds.drop_vars(["x_caa", "x_faa"])

            #+++ Get rid of slight misalignment in times
            ds = adjust_times(ds, round_times=True)
            #---

            #+++ Get specific times and create new variables
            t_slice = slice(ds.T_advective_spinup+10, np.inf, 1)
            ds = ds.sel(time=t_slice)
            ds = ds.assign_coords(time=ds.time-ds.time[0])

            #+++ Get a unique list of time (make it as long as the longest ds
            if not dslist:
                time_values = np.array(ds.time)
            else:
                if len(np.array(ds.time)) > len(time_values):
                       time_values = np.array(ds.time)
            #---
            #---
        #---
        #---

        #+++ Calculate resolutions before they get thrown out
        if "Δx_min" not in ds.keys(): ds["Δx_min"] = ds["Δx_caa"].where(ds["Δx_caa"] > 0).min().values
        if "Δy_min" not in ds.keys(): ds["Δy_min"] = ds["Δy_aca"].where(ds["Δy_aca"] > 0).min().values
        if "Δz_min" not in ds.keys(): ds["Δz_min"] = ds["Δz_aac"].where(ds["Δz_aac"] > 0).min().values
        #---

        #+++ Create auxiliary variables and organize them into a Dataset
        if "PV" in ds.variables.keys():
            ds["PV_norm"] = ds.PV / (ds.N2_inf * ds.f_0)
        ds["simulation"] = simname
        ds["sim_number"] = sim_number
        ds["f₀"] = ds.f_0
        ds["N²∞"] = ds.N2_inf
        ds = ds.expand_dims(("Ro_h", "Fr_h")).assign_coords(Ro_h=[np.round(ds.Ro_h, decimals=4)],
                                                            Fr_h=[np.round(ds.Fr_h, decimals=4)])
        dslist.append(ds)
        #---

    #+++ Create snapshots dataset
    for i, ds in enumerate(dslist[1:]):
        try:
            if slice_name == "xyii":
                assert np.allclose(dslist[0].y_aca.values, ds.y_aca.values), "y coordinates don't match in all datasets"
            elif slice_name == "xiza":
                assert np.allclose(dslist[0].z_aac.values, ds.z_aac.values), "z coordinates don't match in all datasets"
        except AttributeError:
            if slice_name == "xyii":
                assert np.allclose(dslist[0].y.values, ds.y.values), "y coordinates don't match in all datasets"
            elif slice_name == "xiza":
                assert np.allclose(dslist[0].z.values, ds.z.values), "z coordinates don't match in all datasets"
        if "time" in ds.coords.keys():
            if verbose:
                for ds in dslist:
                    print(ds.time)
            assert np.allclose(dslist[0].time.values, ds.time.values), "Time coordinates don't match in all datasets"

    print("Starting to concatenate everything into one dataset")
    if slice_name != "tafields" and slice_name != "turbstats":
        for i in range(len(dslist)):
            dslist[i]["time"] = time_values # Prevent double time, e.g. [0, 0.2, 0.2, 0.4, 0.4, 0.6, 0.8] etc. (not sure why this is needed)
    dsout = xr.combine_by_coords(dslist, combine_attrs="drop_conflicts")

    if "Δx_caa" in ds.keys():
        dsout["Δx_caa"] = dsout["Δx_caa"].isel(Ro_h=0, Fr_h=0)
        dsout["land_mask"]  = (dsout["Δx_caa"] == 0)
        dsout["water_mask"] = np.logical_not(dsout.land_mask)
    if "Δy_aca" in ds.keys(): dsout["Δy_aca"] = dsout["Δy_aca"].isel(Ro_h=0, Fr_h=0)
    if "Δz_aac" in ds.keys(): dsout["Δz_aac"] = dsout["Δz_aac"].isel(Ro_h=0, Fr_h=0)
    #---

    return dsout
#---

#+++ Dataset utility functions
def integrate(da, dV=None, dims=("x", "y", "z")):
    """
    Integrate a data array over specified dimensions using volume elements.

    Parameters
    ----------
    da : xarray.DataArray
        The data array to integrate
    dV : xarray.DataArray, optional
        Volume elements for integration. If None, will be constructed from grid spacing.
    dims : tuple of str, optional
        Dimensions to integrate over. Default ("x", "y", "z")

    Returns
    -------
    xarray.DataArray
        The integrated result
    """
    if dV is None:
        # This function assumes the dataset has the required grid spacing variables
        # It should be called in a context where these are available
        raise ValueError("dV must be provided when calling integrate() outside of a dataset context")
    return (da * dV).pnsum(dims)

def drop_faces(ds, drop_coords=True):
    """
    Drop all variables that have face coordinates (x_faa, y_afa, z_aaf)

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to filter
    drop_coords : bool, optional
        Whether to also drop coordinates with face dimensions. Default True.

    Returns
    -------
    xarray.Dataset
        Dataset with face variables removed
    """
    face_dims = ["x_faa", "y_afa", "z_aaf"]

    # Get variables to drop
    vars_to_drop = []
    for var in ds.variables:
        if any(dim in ds[var].dims for dim in face_dims):
            if var in ds.coords and not drop_coords:
                continue
            vars_to_drop.append(var)

    return ds.drop_vars(vars_to_drop)

def mask_immersed(da, bathymetric_mask):
    """
    Mask data array using bathymetric mask.

    Parameters
    ----------
    da : xarray.DataArray
        Data array to mask
    bathymetric_mask : xarray.DataArray
        Bathymetric mask (True where masked, False where valid)

    Returns
    -------
    xarray.DataArray
        Masked data array
    """
    return da.where(np.logical_not(bathymetric_mask))
#---

#+++ Downsample / chunk
def down_chunk(ds, max_time=np.inf, **kwargs):
    ds = ds.sel(time=slice(0, max_time))
    ds = pn.downsample(ds, **kwargs)
    ds = ds.pnchunk(maxsize_4d=1000**2, round_func=np.ceil)
    return ds
#---

#+++ Transform attributes to variables
def gather_attributes_as_variables(ds, ds_ref=None, include_derived=True):
    """
    Transform dataset attributes to variables and optionally create derived quantities.

    This function converts common attributes like Ro_h, Fr_h, etc. to variables
    in the dataset, and optionally creates derived quantities like RoFr, V∞³÷L, etc.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to transform
    xyza : xarray.Dataset, optional
        Reference dataset containing grid spacing variables (Δx_caa, Δy_aca, Δz_aac)
        for calculating minimum resolutions. If None, will use ds itself.
    include_derived : bool, optional
        Whether to include derived quantities. Default True.

    Returns
    -------
    xarray.Dataset
        Dataset with attributes converted to variables and optional derived quantities
    """
    # Use ds as reference if xyza not provided
    if ds_ref is None:
        ds_ref = ds

    #+++ Convert existing variables to new names and attributes to variables
    for var in ["Ro_h", "Fr_h", "Slope_Bu", "α", "Bu_h", "Γ", "c_dz",
                "f₀", "N²∞", "V∞", "L"]:
        if var in ds.variables:
            ds[var] = ds[var]
        elif var in ds.attrs:
            ds[var] = ds.attrs[var]
    #---

    #+++ Calculate minimum resolutions
    if "Δx_caa" in ds_ref.variables:
        ds["Δx_min"] = ds_ref["Δx_caa"].min()
    if "Δy_aca" in ds_ref.variables:
        ds["Δy_min"] = ds_ref["Δy_aca"].min()
    if "Δz_aac" in ds_ref.variables:
        ds["Δz_min"] = ds_ref["Δz_aac"].min()
    #---

    #+++ Create derived quantities if requested
    if include_derived:
        # Basic derived quantities
        if "Ro_h" in ds.variables and "Fr_h" in ds.variables:
            ds["RoFr"] = ds.Ro_h * ds.Fr_h

        # Velocity and stratification derived quantities
        if "V∞" in ds.variables and "L" in ds.variables:
            ds["V∞³÷L"] = ds["V∞"]**3 / ds.L

        if "V∞" in ds.variables and "N²∞" in ds.variables:
            ds["V∞²N∞"] = ds["V∞"]**2 * np.sqrt(ds["N²∞"])

        if "N²∞" in ds.variables and "L" in ds.variables:
            ds["N∞³L²"] = np.sqrt(ds["N²∞"])**3 * ds.L**2
    #---

    return ds
#---
