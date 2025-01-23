@info "Starting to plot video..."

#+++ Figure out if we have a screen or not (https://github.com/JuliaPlots/Plots.jl/issues/4368)
if ("GITHUB_ENV" ∈ keys(ENV)) || # Is it a github CI?
    ("PBS_JOBID" ∈ keys(ENV)) # Is it a PBS job?
    @info "Headless server! (probably NCAR or github CI). Loading CairoMakie"
    using CairoMakie
    Mk = CairoMakie
    get!(ENV, "GKSwstype",  "nul")
    Mk = CairoMakie
else
    @info "Loading GLMakie"
    using GLMakie
    Mk = GLMakie
end
@info "Backend loaded"
#---

#+++ Read datasets
if @isdefined simulation
    fpath_xyi = simulation.output_writers[:nc_xyi].filepath
else
    simname = "tokara-f64"
    fpath_xyi = "data/xyi.$simname.nc"
end

using Rasters
using Rasters: name
import NCDatasets

function squeeze(ds::Union{Raster, RasterStack})
    flat_dimensions = NamedTuple((name(dim), 1) for dim in dims(ds) if length(dims(ds, dim)) ==  1)
    return getindex(ds; flat_dimensions...)
end

broad_variables = (:v, :PV, :εₖ, :Ro,)
@info "Reading ds_xyi"
ds_xyi = RasterStack(fpath_xyi, lazy=true, name=broad_variables)

#+++ Get rid of extra time steps from picking up simulations
using StatsBase
times_orig = dims(ds_xyi, :Ti)
Δt = mode(times_orig[2:end] .- times_orig[1:end-1])
times_fixed = 0:Δt:times_orig[end]
ds_xyi = ds_xyi[Ti=Near(times_fixed)]
#---

# Get other datasets
dslist = Vector{Any}([(ds_xyi, "xy")])
for (prefix, slice) in [("xiz", "xz"), ("iyz", "yz")]
    fpath = replace(fpath_xyi, "xyi" => prefix)
    if isfile(fpath)
        ds = RasterStack(fpath, lazy=true, name=broad_variables)
        push!(dslist, (ds[Ti=Near(times_fixed)], slice))
    end
end

# Get the indices for the slices
slicelist = []
for (i, (ds, slice)) in enumerate(dslist)
    dim_index = if slice == "xy"
                    :zC
                elseif slice == "xz"
                    :yC
                elseif slice == "yz"
                    :xC
                elseif slice == "xyz"
                    nothing
                end
    dim_value = dims(ds, dim_index)[1]
    push!(slicelist, (slice, string(first(string(dim_index))), dim_value))
end
#---

#+++ Get parameters
if !((@isdefined params) && (@isdefined simulation))
    md = metadata(ds_xyi)
    params = (; (Symbol(k) => v for (k, v) in md)...)
end
#---

#+++ Auxiliary parameters
u_lims = (-params.V∞, +params.V∞) .* 1.2
w_lims = u_lims
PV_lims = params.N²∞ * abs(params.f₀) * [-5, +5]
ε_max = maximum(ds_xyi.εₖ)
ε_lims = (ε_max/1e6, ε_max/1e2)
#---

#+++ Decide datasets, frames, etc.
times = dims(ds_xyi, :Ti)
n_times = length(times)
max_frames = 200
step = max(1, floor(Int, n_times / max_frames))

dslist = [ (squeeze(ds), slice) for (ds, slice) in dslist ]
#---

#+++ Plotting options
variables = broad_variables

kwargs = Dict(:u => (colorrange = u_lims,
                     colormap = :balance),
              :v => (colorrange = u_lims,
                     colormap = :balance),
              :w => (colorrange = w_lims,
                     colormap = :balance),
              :PV => (colorrange = PV_lims,
                      colormap = :seismic),
              :εₖ => (colormap = :inferno,
                      colorscale = log10,
                      colorrange = ε_lims,),
              :Ro => (; colorrange = (-2, +2),
                      colormap = :balance),
              :Ri => (; colorrange = (-2, +2),
                      colormap = :balance),
              )

title_height = 8
panel_height = 140; panel_width = 300
cbar_height = 8
bottom_axis_height = 2panel_height/3
#---

#+++ Plotting preamble
using Oceananigans.Units, Printf
using Oceananigans: prettytime

fig = Figure(resolution = (1500, 500))
n = Observable(1)

title = @lift "α = $(@sprintf "%.2g" params.α),     Frₕ = $(@sprintf "%.2g" params.Fr_h),    Roₕ = $(@sprintf "%.2g" params.Ro_h);    " *
              "Sᴮᵘ = $(@sprintf "%.2g" params.Slope_Bu);    " *
              "V∞ = $(@sprintf "%.2g" params.V∞) m/s,    Δzₘᵢₙ = $(@sprintf "%.2g" params.Δz_min) m,    z₀ = $(@sprintf "%.2g" params.z₀) m;     " *
              "Time = $(@sprintf "%s" prettytime(times[$n]))  =  $(@sprintf "%.2g" times[$n]/params.T_advective) advective periods  =  " *
              "$(@sprintf "%.2g" times[$n]/params.T_inertial) Inertial periods"
fig[1, 1:length(variables)] = Label(fig, title, fontsize=18, tellwidth=false, height=title_height)

dimnames_tup = (:xF, :xC, :yF, :yC, :zF, :zC)
#---

#+++ Create axes and populate them
for (i, variable) in enumerate(variables)
    for (j, (ds, slice)) in enumerate(dslist)
        @info "Setting up $variable panel with i=$i j=$j"

        global var_raster = ds[variable]
        dimnames = collect( el for el in dimnames_tup if el in map(name, dims(var_raster)) )
        push!(dimnames, :Ti)
        @show dimnames

        v = permutedims(var_raster, dimnames)
        vₙ = @lift v[Ti=$n]

        #+++ Set axes labels
        panel_title = j == 1            ? string(variable)      : ""
        ylabel      = i == 1            ? string(dimnames[2])   : ""
        #---

        #+++ Create axis and plot heatmap
        height = slice == "xy" ? 2*panel_width : panel_width/2
        ax = Axis(fig[j+1, i]; title=panel_title, xlabel=string(dimnames[1]), ylabel, width=panel_width, height)
        global hm = heatmap!(vₙ; kwargs[variable]...)
        #---

        #+++ Plot vlines when appropriate
        for (other_slice, dim, dim_value) in slicelist
            if dim == string(first(string(dimnames[1])))
                vlines!(ax, dim_value, color=:white, linestyle=:dash)
            end
        end
        #---

        #+++ Plot contours if possible
        try
            b = permutedims(ds[:b], (:xC, :zC, :Ti))
            bₙ = @lift b[:,:,$n]
            contour!(ax, bₙ; levels=10, color=:white, linestyle=:dash, linewidth=0.5)
        catch e
            try
                b = permutedims(ds[:b], (:yC, :zC, :Ti))
                bₙ = @lift b[:,:,$n]
                contour!(ax, bₙ; levels=10, color=:white, linestyle=:dash, linewidth=0.5)
            catch e
            end
        end
        #---
    end

    cbar_label = try metadata(var_raster)["units"] catch e "" end
    Colorbar(fig[length(dslist)+2, i], hm; label=cbar_label, vertical=false, height=cbar_height, ticklabelsize=12)
end
#---

#+++ Record animation
frames = 1:step:n_times
@show step n_times max_frames length(frames)

resize_to_layout!(fig) # Resize figure after everything is done to it, but before recording
Mk.record(fig, "$(@__DIR__)/../anims/$(params.simname).mp4", frames, framerate=14) do frame
    @info "Plotting time step $frame of $(n_times)..."
    n[] = frame
end
#---
