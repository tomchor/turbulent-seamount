@info "Starting to plot video..."

#+++ Setup Makie backend based on environment
is_headless = ("GITHUB_ENV" ∈ keys(ENV)) || ("PBS_JOBID" ∈ keys(ENV))
if is_headless
    @info "Headless environment detected. Loading CairoMakie"
    using CairoMakie
    get!(ENV, "GKSwstype", "nul")
    Mk = CairoMakie
else
    @info "Interactive environment. Loading GLMakie"
    using GLMakie
    Mk = GLMakie
end
#---

#+++ Load re
using Rasters
using Rasters: name  # Need this for name() function
using StatsBase, Printf
using Oceananigans.Units
using Oceananigans: prettytime
import NCDatasets
#---

#+++ Helper functions
function squeeze(ds::Union{Raster, RasterStack})
    flat_dimensions = NamedTuple((name(dim), 1) for dim in dims(ds) if length(dims(ds, dim)) == 1)
    return getindex(ds; flat_dimensions...)
end

# Slice dimension mapping
const SLICE_DIMS = Dict(
    "xy" => :z_aac,
    "xz" => :y_aca,
    "yz" => :x_caa
)
#---

#+++ Read datasets
variables = (:v, :PV, :εₖ, :Ro)

# Get main dataset path
fpath_xyi = if @isdefined simulation
    simulation.output_writers[:nc_xyi].filepath
else
    "data/xyi.seamount.nc"
end

@info "Reading primary dataset: $fpath_xyi"
ds_xyi = RasterStack(fpath_xyi, lazy=true, name=variables)

# Load additional datasets if they exist
datasets = [(ds_xyi, "xy")]
for (prefix, slice_name) in [("xiz", "xz"), ("iyz", "yz")]
    fpath = replace(fpath_xyi, "xyi" => prefix)
    if isfile(fpath)
        ds = RasterStack(fpath, lazy=true, name=variables)
        push!(datasets, (ds, slice_name))
    end
end

# Process datasets and extract slice information
datasets = [(squeeze(ds), slice) for (ds, slice) in datasets]
slice_info = []
for (ds, slice) in datasets
    dim_index = SLICE_DIMS[slice]

    # Check if dimension exists in dataset
    if dim_index in map(name, dims(ds))
        dim_values = dims(ds, dim_index)
        if length(dim_values) > 0
            dim_value = dim_values[1]
            dim_name = string(first(string(dim_index)))
            push!(slice_info, (slice, dim_name, dim_value))
        else
            @warn "Dimension $dim_index exists but is empty in dataset $slice"
        end
    else
        @warn "Dimension $dim_index not found in dataset $slice, skipping slice indicator"
    end
end
#---

#+++ Get parameters
if !((@isdefined params) && (@isdefined simulation))
    md = metadata(ds_xyi)
    params = (; (Symbol(k) => v for (k, v) in md)...)
end
#---

#+++ Setup animation parameters
times = dims(ds_xyi, :Ti)
n_times = length(times)
max_frames = 200
frame_step = max(1, floor(Int, n_times / max_frames))
frames = 1:frame_step:n_times

@info "Animation setup: $n_times time steps → $(length(frames)) frames (step = $frame_step)"
#---

#+++ Define plotting parameters
# Color ranges for each variable
color_ranges = Dict(
    :v  => (range=(-params.V∞, +params.V∞) .* 1.2, colormap=:balance),
    :PV => (range=params.N²∞ * abs(params.f₀) * [-5, +5], colormap=:seismic),
    :εₖ => (range=(1e-10, 1e-7), colormap=:inferno, colorscale=log10),
    :Ro => (range=(-4, +4), colormap=:balance)
)

# Layout parameters
layout_params = (
    title_height = 10,
    panel_width = 300,
    cbar_height = 8,
    column_gap = 0
)
#---

#+++ Create figure and setup
fig = Figure(figure_padding = (10, 30, 10, 10))
n = Observable(1)

# Create title
title = @lift "α = $(@sprintf "%.2g" params.α),     Frₕ = $(@sprintf "%.2g" params.Fr_h),    Roₕ = $(@sprintf "%.2g" params.Ro_h);    Sᴮᵘ = $(@sprintf "%.2g" params.Slope_Bu);    " *
              "Δzₘᵢₙ = $(@sprintf "%.2g" params.Δz_min) m,    Time = $(@sprintf "%s" prettytime(times[$n]))  =  $(@sprintf "%.3g" times[$n]/params.T_advective) advective periods  =  " *
              "$(@sprintf "%.3g" times[$n]/params.T_inertial) Inertial periods"

fig[1, 1:length(variables)] = Label(fig, title, fontsize=18, tellwidth=false, height=layout_params.title_height)
colgap!(fig.layout, layout_params.column_gap)
#---

#+++ Create axes and plots
dimnames_order = (:x_faa, :x_caa, :y_afa, :y_aca, :z_afa, :z_aac)

for (i, variable) in enumerate(variables)
    for (j, (ds, slice)) in enumerate(datasets)
        @info "Creating panel: $variable ($slice)"

        # Get variable data and determine dimensions
        var_data = ds[variable]
        dimnames = [dim for dim in dimnames_order if dim in map(name, dims(var_data))]
        push!(dimnames, :Ti)

        # Permute dimensions and create observable
        v = permutedims(var_data, dimnames)
        vₙ = @lift v[Ti=$n]

        # Set up axis properties
        panel_title = j == 1 ? string(variable) : ""
        ylabel = i == 1 ? string(dimnames[2]) : ""
        xlabel = string(dimnames[1])
        height = slice == "xy" ? 2 * layout_params.panel_width : layout_params.panel_width ÷ 2

        # Create axis
        ax = Axis(fig[j+1, i];
                  title=panel_title, xlabel, ylabel,
                  width=layout_params.panel_width, height)

        # Hide y-axis decorations for non-leftmost panels
        if i > 1
            hideydecorations!(ax, label=false, ticklabels=true, ticks=false, grid=false)
        end

        # Create heatmap
        color_params = color_ranges[variable]
        global hm = heatmap!(vₙ; colorrange=color_params.range, colormap=color_params.colormap,
                            (haskey(color_params, :colorscale) ? (colorscale=color_params.colorscale,) : ())...)

        # Add slice indicator lines
        for (other_slice, dim_name, dim_value) in slice_info
            if dim_name == string(first(xlabel))
                vlines!(ax, dim_value, color=:white, linestyle=:dash, alpha=0.7)
            end
        end

        # Add buoyancy contours if available
        if haskey(ds, :b)
            for dim_combo in [(:x_caa, :z_aac), (:y_aca, :z_aac)]
                try
                    b = permutedims(ds[:b], (dim_combo..., :Ti))
                    bₙ = @lift b[:, :, $n]
                    contour!(ax, bₙ; levels=10, color=:white, linestyle=:dash, linewidth=0.5, alpha=0.6)
                    break
                catch
                    continue
                end
            end
        end
    end

    # Add colorbar
    cbar_label = try
        metadata(datasets[1][1][variable])["units"]
    catch
        string(variable)
    end

    Colorbar(fig[length(datasets)+2, i], hm;
             label=cbar_label, vertical=false,
             height=layout_params.cbar_height, ticklabelsize=12)
end
#---

#+++ Record animation
@info "Recording animation with $(length(frames)) frames"
resize_to_layout!(fig)

Mk.record(fig, "$(@__DIR__)/../anims/$(params.simname).mp4", frames,
         framerate=14, compression=30, px_per_unit=1) do frame
    @info "Frame $frame / $(frames[end])"
    n[] = frame
end

@info "Animation saved successfully!"
#---
