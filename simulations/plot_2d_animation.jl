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

#+++ Load required packages
using Printf: @sprintf
using Oceananigans
using Oceananigans.Units
using Oceananigans: prettytime
import NCDatasets
#---

#+++ Read datasets
variables = (:u, :PV, :εₖ, :Ro)

# Get main dataset paths
simname_fallback = "balanus_Ro_b0.05_Fr_b0.3_L0.8_dz4_T_adv_spinup12"
fpath_xyii = (@isdefined simulation) ? simulation.output_writers[:nc_xyii].filepath : "data/xyii.$simname_fallback.nc"
fpath_xizi = (@isdefined simulation) ? simulation.output_writers[:nc_xizi].filepath : "data/xizi.$simname_fallback.nc"

@info "Reading xyii dataset: $fpath_xyii"
# Load each variable as a FieldTimeSeries
timeseries_xyii = Dict(var => FieldTimeSeries(fpath_xyii, String(var),) for var in variables)

@info "Reading xizi dataset: $fpath_xizi"
timeseries_xizi = Dict(var => FieldTimeSeries(fpath_xizi, String(var)) for var in variables)

# Also load buoyancy if available for contours
try
    timeseries_xyii[:b] = FieldTimeSeries(fpath_xyii, "b")
    timeseries_xizi[:b] = FieldTimeSeries(fpath_xizi, "b")
catch e
    @warn "Could not load buoyancy field 'b': $e"
end
#---

#+++ Get parameters
if !((@isdefined params) && (@isdefined simulation))
    # Read metadata from NetCDF file
    NCDatasets.NCDataset(fpath_xyii) do ds
        params = (; (Symbol(k) => ds.attrib[k] for k in keys(ds.attrib))...)
    end
end
#---

#+++ Setup animation parameters
# Get times from the first FieldTimeSeries
times = timeseries_xyii[first(variables)].times
n_times = length(times)
max_frames = 200
frame_step = max(1, floor(Int, n_times / max_frames))
frames = 1:frame_step:n_times

@info "Animation setup: $n_times time steps → $(length(frames)) frames (step = $frame_step)"
#---

#+++ Define plotting parameters
# Color ranges for each variable
color_ranges = Dict(
    :u  => (range=(-params.U∞, +params.U∞) .* 1.2, colormap=:balance),
    :v  => (range=(-params.U∞, +params.U∞) .* 1.2, colormap=:balance),
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
title = @lift "α = $(@sprintf "%.2g" params.α), Frₕ = $(@sprintf "%.2g" params.Fr_b), Roₕ = $(@sprintf "%.2g" params.Ro_b), Sᴮᵘ = $(@sprintf "%.2g" params.Slope_Bu), Δz = $(@sprintf "%.2g" params.Δz_min) m;   Time = $(@sprintf "%s" prettytime(times[$n])) = $(@sprintf "%.3g" times[$n]/params.T_adv) adv periods = $(@sprintf "%.3g" times[$n]/params.T_inertial) Inertial periods"

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
for (i, variable) in enumerate(variables)
    @info "Creating panel: $variable"

    # Get FieldTimeSeries for this variable
    field_xyii = timeseries_xyii[variable]
    field_xizi = timeseries_xizi[variable]

    # Create observables for current time step
    fieldₙ_xyii = @lift field_xyii[$n]
    fieldₙ_xizi = @lift field_xizi[$n]

    # Calculate data aspect ratio from grid
    grid_xyii = field_xyii.grid
    dims_xyii = size(field_xyii.grid)
    # Estimate aspect ratio (first two dimensions)
    aspect_ratio_xyii = dims_xyii[1] / dims_xyii[2]

    # Set panel dimensions
    panel_width = layout_params.panel_width
    panel_height_xyii = panel_width / aspect_ratio_xyii
    panel_height_xyii = clamp(panel_height_xyii, panel_width * 0.3, panel_width * 2.0)

    # Determine axis labels from grid and field location
    # xyii is typically x-y plane, xizi is typically x-z plane
    xlabel_xyii = "x (m)"
    ylabel_xyii = "y (m)"
    xlabel_xizi = "x (m)"
    ylabel_xizi = "z (m)"

    # Create xyii axis
    if i == length(variables)
        # Bottom panel: show x label
        ax_xyii = Axis(fig[i+2, 1];
                      xlabel=xlabel_xyii, ylabel=ylabel_xyii,
                      width=panel_width, height=panel_height_xyii)
    else
        # Upper panels: no x label
        ax_xyii = Axis(fig[i+2, 1];
                       ylabel=ylabel_xyii,
                       width=panel_width, height=panel_height_xyii)

        # Hide all x decorations for upper panels
        hidexdecorations!(ax_xyii, label=false, ticklabels=false, ticks=false, grid=false)
        ax_xyii.xticks = (Float64[], String[])
        ax_xyii.xticklabelsize = 0
        ax_xyii.xticksize = 0
    end

    # Create xyii heatmap
    color_params = color_ranges[variable]
    global hm_xyii = heatmap!(ax_xyii, fieldₙ_xyii; 
                              colorrange=color_params.range, 
                              colormap=color_params.colormap,
                              (haskey(color_params, :colorscale) ? (colorscale=color_params.colorscale,) : ())...)

    # Add buoyancy contours to xyii if available
    if haskey(timeseries_xyii, :b)
        try
            b_field_xyii = timeseries_xyii[:b]
            bₙ_xyii = @lift b_field_xyii[$n]
            contour!(ax_xyii, bₙ_xyii; levels=10, color=:white, linestyle=:dash, linewidth=0.5, alpha=0.6)
        catch e
            @warn "Could not add buoyancy contours to xyii: $e"
        end
    end

    # Use the same panel height for xizi for consistent layout
    panel_height_xizi = panel_height_xyii

    # Create xizi axis
    if i == length(variables)
        # Bottom panel: show x label
        ax_xizi = Axis(fig[i+2, 2];
                       xlabel=xlabel_xizi, ylabel=ylabel_xizi,
                       width=panel_width, height=panel_height_xizi)
    else
        # Upper panels: no x label
        ax_xizi = Axis(fig[i+2, 2];
                       ylabel=ylabel_xizi,
                       width=panel_width, height=panel_height_xizi)

        # Hide all x decorations for upper panels
        hidexdecorations!(ax_xizi, label=false, ticklabels=false, ticks=false, grid=false)
        ax_xizi.xticks = (Float64[], String[])
        ax_xizi.xticklabelsize = 0
        ax_xizi.xticksize = 0
    end

    # Create xizi heatmap
    global hm_xizi = heatmap!(ax_xizi, fieldₙ_xizi; 
                              colorrange=color_params.range, 
                              colormap=color_params.colormap,
                              (haskey(color_params, :colorscale) ? (colorscale=color_params.colorscale,) : ())...)

    # Add buoyancy contours to xizi if available
    if haskey(timeseries_xizi, :b)
        try
            b_field_xizi = timeseries_xizi[:b]
            bₙ_xizi = @lift b_field_xizi[$n]
            contour!(ax_xizi, bₙ_xizi; levels=10, color=:white, linestyle=:dash, linewidth=0.5, alpha=0.6)
        catch e
            @warn "Could not add buoyancy contours to xizi: $e"
        end
    end

    # Add colorbar (column 3)
    cbar_label = String(variable)

    # Use the same height for the colorbar since both panels have the same height
    Colorbar(fig[i+2, 3], hm_xyii;
             label=cbar_label, vertical=true,
             width=layout_params.cbar_height, height=panel_height_xyii, ticklabelsize=12)
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