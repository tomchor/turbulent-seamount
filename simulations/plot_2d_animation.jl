@info "Starting to plot video..."

#+++ Setup Makie backend based on environment
is_headless() = !haskey(ENV, "DISPLAY") || isempty(ENV["DISPLAY"])
has_opengl() = try !isempty(read(`glxinfo -B`, String)) catch; false end
if is_headless() || !has_opengl()
    @info "Either headless environment, or environment without openGL detected. Loading CairoMakie"
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
import Rasters as ra
import NCDatasets
using Rasters: Raster, RasterStack
using Printf: @sprintf
using Oceananigans.Units
using Oceananigans: prettytime
#---

#+++ Helper functions
function squeeze(ds::Union{Raster, RasterStack})
    flat_dimensions = NamedTuple((ra.name(dim), 1) for dim in ra.dims(ds) if length(ra.dims(ds, dim)) == 1)
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
fpath_xyii = (@isdefined simulation) ? simulation.output_writers[:nc_xyii].filepath : "data/xyii.seamount_Ro_h0.2_Fr_h1.25_L0_FWHM300_dz8.nc"

@info "Reading primary dataset: $fpath_xyii"
ds_xyii = RasterStack(fpath_xyii, lazy=true, name=variables)
ds_xyii = squeeze(ds_xyii)
#---

#+++ Get parameters
if !((@isdefined params) && (@isdefined simulation))
    md = ra.metadata(ds_xyii)
    params = (; (Symbol(k) => v for (k, v) in md)...)
end
#---

#+++ Setup animation parameters
times = ra.dims(ds_xyii, :Ti)
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
    column_gap = 0,
    row_gap = 10
)
#---

#+++ Create figure and setup
fig = Figure(figure_padding = (10, 30, 10, 10))
n = Observable(1)

# Create title
title = @lift "α = $(@sprintf "%.2g" params.α),     Frₕ = $(@sprintf "%.2g" params.Fr_h),    Roₕ = $(@sprintf "%.2g" params.Ro_h);    Sᴮᵘ = $(@sprintf "%.2g" params.Slope_Bu);    " *
              "Δzₘᵢₙ = $(@sprintf "%.2g" params.Δz_min) m,    Time = $(@sprintf "%s" prettytime(times[$n]))  =  $(@sprintf "%.3g" times[$n]/params.T_advective) advective periods  =  " *
              "$(@sprintf "%.3g" times[$n]/params.T_inertial) Inertial periods"

fig[1, 1] = Label(fig, title, fontsize=18, tellwidth=false, height=layout_params.title_height)
colgap!(fig.layout, layout_params.column_gap)
rowgap!(fig.layout, layout_params.row_gap)
#---

#+++ Create axes and plots
dimnames_order = (:x_faa, :x_caa, :y_afa, :y_aca, :z_afa, :z_aac)

for (i, variable) in enumerate(variables)
    @info "Creating panel: $variable"
    
    # Get variable data and determine dimensions
    var_data = ds_xyii[variable]
    dimnames = [dim for dim in dimnames_order if dim in map(ra.name, ra.dims(var_data))]
    push!(dimnames, :Ti)
    
    # Permute dimensions and create observable
    v = permutedims(var_data, dimnames)
    vₙ = @lift v[Ti=$n]
    
    # Set up axis properties
    panel_title = string(variable)
    ylabel = string(dimnames[2])
    xlabel = string(dimnames[1])
    
    # Calculate data aspect ratio
    data_dims = size(v)
    aspect_ratio = data_dims[1] / data_dims[2]  # width / height
    
    # Set panel dimensions based on data aspect ratio with reasonable bounds
    panel_width = layout_params.panel_width
    panel_height = panel_width / aspect_ratio
    
    # Apply reasonable bounds to prevent extremely tall or wide panels
    min_height = layout_params.panel_width * 0.3  # Minimum height = 30% of width
    max_height = layout_params.panel_width * 2.0  # Maximum height = 200% of width
    panel_height = clamp(panel_height, min_height, max_height)
    
    @info "Variable $variable: data dimensions = $data_dims, aspect ratio = $(@sprintf "%.2f" aspect_ratio), panel size = $(@sprintf "%.0f" panel_width) × $(@sprintf "%.0f" panel_height)"
    
    # Create axis
    ax = Axis(fig[i+1, 1];
              title=panel_title, xlabel, ylabel,
              width=panel_width, height=panel_height)
    
    # Create heatmap
    color_params = color_ranges[variable]
    global hm = heatmap!(vₙ; colorrange=color_params.range, colormap=color_params.colormap,
                        (haskey(color_params, :colorscale) ? (colorscale=color_params.colorscale,) : ())...)
    
    # Add buoyancy contours if available
    if haskey(ds_xyii, :b)
        for dim_combo in [(:x_caa, :z_aac), (:y_aca, :z_aac)]
            try
                b = permutedims(ds_xyii[:b], (dim_combo..., :Ti))
                bₙ = @lift b[:, :, $n]
                contour!(ax, bₙ; levels=10, color=:white, linestyle=:dash, linewidth=0.5, alpha=0.6)
                break
            catch
                continue
            end
        end
    end
    
    # Add colorbar
    cbar_label = try
        metadata(ds_xyii[variable])["units"]
    catch
        string(variable)
    end
    
    Colorbar(fig[i+1, 2], hm;
             label=cbar_label, vertical=true,
             width=layout_params.cbar_height, height=panel_height, ticklabelsize=12)
end
#---
pause

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
