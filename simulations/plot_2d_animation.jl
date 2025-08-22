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

# Get main dataset paths
fpath_xyii = (@isdefined simulation) ? simulation.output_writers[:nc_xyii].filepath : "data/xyii.seamount_Ro_h0.2_Fr_h1.25_L0_FWHM300_dz8.nc"
fpath_xizi = (@isdefined simulation) ? simulation.output_writers[:nc_xizi].filepath : "data/xizi.seamount_Ro_h0.2_Fr_h1.25_L0_FWHM300_dz8.nc"

@info "Reading xyii dataset: $fpath_xyii"
ds_xyii = RasterStack(fpath_xyii, lazy=true, name=variables)
ds_xyii = squeeze(ds_xyii)

@info "Reading xizi dataset: $fpath_xizi"
ds_xizi = RasterStack(fpath_xizi, lazy=true, name=variables)
ds_xizi = squeeze(ds_xizi)
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
    title_height = 8,
    panel_width = 300,
    cbar_height = 8,
    column_gap = 20,
    row_gap = -50,
    title_row_gap = -50
)
#---

#+++ Create figure and setup
fig = Figure(figure_padding = (10, 30, 10, 10))
n = Observable(1)

# Create title in two lines within one row
title = @lift "α = $(@sprintf "%.2g" params.α), Frₕ = $(@sprintf "%.2g" params.Fr_h), Roₕ = $(@sprintf "%.2g" params.Ro_h), Sᴮᵘ = $(@sprintf "%.2g" params.Slope_Bu), Δz = $(@sprintf "%.2g" params.Δz_min) m;   Time = $(@sprintf "%s" prettytime(times[$n])) = $(@sprintf "%.3g" times[$n]/params.T_advective) adv periods = $(@sprintf "%.3g" times[$n]/params.T_inertial) Inertial periods"

# Create single title row with two lines
fig[1, 1:3] = Label(fig, title, fontsize=18, tellwidth=false, height=layout_params.title_height)

colgap!(fig.layout, layout_params.column_gap)
rowgap!(fig.layout, layout_params.row_gap)

# Configure column widths: xyii plot column wide, xizi plot column wide, colorbar column narrow
colsize!(fig.layout, 1, Auto(1.0))  # xyii plot column (main width)
colsize!(fig.layout, 2, Auto(1.0))  # xizi plot column (main width)
colsize!(fig.layout, 3, Auto(0.6))  # Colorbar column (wider for better spacing)
#---

#+++ Create axes and plots
dimnames_order = (:x_faa, :x_caa, :y_afa, :y_aca, :z_afa, :z_aac)

for (i, variable) in enumerate(variables)
    @info "Creating panel: $variable"
    
    # Create xyii plot (column 1)
    var_data_xyii = ds_xyii[variable]
    dimnames_xyii = [dim for dim in dimnames_order if dim in map(ra.name, ra.dims(var_data_xyii))]
    push!(dimnames_xyii, :Ti)
    
    # Permute dimensions and create observable for xyii
    v_xyii = permutedims(var_data_xyii, dimnames_xyii)
    v_xyiiₙ = @lift v_xyii[Ti=$n]
    
    # Calculate data aspect ratio for xyii
    data_dims_xyii = size(v_xyii)
    aspect_ratio_xyii = data_dims_xyii[1] / data_dims_xyii[2]
    
    # Set panel dimensions for xyii
    panel_width = layout_params.panel_width
    panel_height_xyii = panel_width / aspect_ratio_xyii
    panel_height_xyii = clamp(panel_height_xyii, panel_width * 0.3, panel_width * 2.0)
    
    # Create xyii axis
    if i == length(variables)
        # Bottom panel: show x label
        ax_xyii = Axis(fig[i+2, 1];
                      xlabel=string(dimnames_xyii[1]), ylabel=string(dimnames_xyii[2]),
                      width=panel_width, height=panel_height_xyii)
    else
        # Upper panels: no x label
        ax_xyii = Axis(fig[i+2, 1];
                      ylabel=string(dimnames_xyii[2]),
                      width=panel_width, height=panel_height_xyii)
        
        # Hide all x decorations for upper panels
        hidexdecorations!(ax_xyii, label=false, ticklabels=false, ticks=false, grid=false)
        ax_xyii.xticks = (Float64[], String[])
        ax_xyii.xticklabelsize = 0
        ax_xyii.xticksize = 0
    end
    
    # Create xyii heatmap
    color_params = color_ranges[variable]
    global hm_xyii = heatmap!(v_xyiiₙ; colorrange=color_params.range, colormap=color_params.colormap,
                              (haskey(color_params, :colorscale) ? (colorscale=color_params.colorscale,) : ())...)
    
    # Add buoyancy contours to xyii if available
    if haskey(ds_xyii, :b)
        for dim_combo in [(:x_caa, :z_aac), (:y_aca, :z_aac)]
            try
                b = permutedims(ds_xyii[:b], (dim_combo..., :Ti))
                bₙ = @lift b[:, :, $n]
                contour!(ax_xyii, bₙ; levels=10, color=:white, linestyle=:dash, linewidth=0.5, alpha=0.6)
                break
            catch
                continue
            end
        end
    end
    
    # Create xizi plot (column 2)
    var_data_xizi = ds_xizi[variable]
    dimnames_xizi = [dim for dim in dimnames_order if dim in map(ra.name, ra.dims(var_data_xizi))]
    push!(dimnames_xizi, :Ti)
    
    # Permute dimensions and create observable for xizi
    v_xizi = permutedims(var_data_xizi, dimnames_xizi)
    v_xiziₙ = @lift v_xizi[Ti=$n]
    
    # Use the same panel height as xyii for consistent layout
    panel_height_xizi = panel_height_xyii
    
    # Create xizi axis
    if i == length(variables)
        # Bottom panel: show x label
        ax_xizi = Axis(fig[i+2, 2];
                      xlabel=string(dimnames_xizi[1]), ylabel=string(dimnames_xizi[2]),
                      width=panel_width, height=panel_height_xizi)
    else
        # Upper panels: no x label
        ax_xizi = Axis(fig[i+2, 2];
                      ylabel=string(dimnames_xizi[2]),
                      width=panel_width, height=panel_height_xizi)
        
        # Hide all x decorations for upper panels
        hidexdecorations!(ax_xizi, label=false, ticklabels=false, ticks=false, grid=false)
        ax_xizi.xticks = (Float64[], String[])
        ax_xizi.xticklabelsize = 0
        ax_xizi.xticksize = 0
    end
    
    # Create xizi heatmap
    global hm_xizi = heatmap!(v_xiziₙ; colorrange=color_params.range, colormap=color_params.colormap,
                              (haskey(color_params, :colorscale) ? (colorscale=color_params.colorscale,) : ())...)
    
    # Add buoyancy contours to xizi if available
    if haskey(ds_xizi, :b)
        for dim_combo in [(:x_caa, :z_aac), (:y_aca, :z_aac)]
            try
                b = permutedims(ds_xizi[:b], (dim_combo..., :Ti))
                bₙ = @lift b[:, :, $n]
                contour!(ax_xizi, bₙ; levels=10, color=:white, linestyle=:dash, linewidth=0.5, alpha=0.6)
                break
            catch
                continue
            end
        end
    end
    
    # Add colorbar (column 3)
    cbar_label = try
        metadata(ds_xyii[variable])["units"]
    catch
        string(variable)
    end
    
    # Use the same height for the colorbar since both panels have the same height
    Colorbar(fig[i+2, 3], hm_xyii;
             label=cbar_label, vertical=true,
             width=layout_params.cbar_height, height=panel_height_xyii, ticklabelsize=12)
end
#---

#+++ Record animation
@info "Recording animation with $(length(frames)) frames"
resize_to_layout!(fig)

pause
Mk.record(fig, "$(@__DIR__)/../anims/$(params.simname).mp4", frames,
         framerate=14, compression=30, px_per_unit=1) do frame
    @info "Frame $frame / $(frames[end])"
    n[] = frame
end

@info "Animation saved successfully!"
#---
