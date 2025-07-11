using Rasters
import NCDatasets
using GLMakie
using Printf
using Oceananigans: prettytime

# Data file path
fpath_xyzi = (@isdefined simulation) ? simulation.output_writers[:nc_xyzi].filepath : "data/xyzi.seamount_Ro_h=0.2_Fr_h=1.25_L=0_dz=4_closure=DSM.nc"

@info "Reading NetCDF file: $fpath_xyzi"
xyzi = RasterStack(fpath_xyzi, name=(:PV, :bottom_height), lazy=true)

# Get metadata and parameters
md = metadata(xyzi)
params = (; (Symbol(k) => v for (k, v) in md)...)

# Extract grid coordinates
xyzi = xyzi[z_aac = 0..1.1*params.H, y_aca = -Inf..6*params.FWHM]
x_range = extrema(dims(xyzi, :x_caa))
y_range = extrema(dims(xyzi, :y_aca))
z_range = extrema(dims(xyzi, :z_aac))
times = dims(xyzi.PV, :Ti)

# Set PV limits based on data range or physical considerations
interior_PV = params.N²∞ * params.f₀
isovalue_abs = 1.4 * interior_PV  # Adjust this based on data
isorange_abs = isovalue_abs/10
PV_lims = 1.2 .* (-isovalue_abs, +isovalue_abs)

# 3D plot settings
Lx = diff(x_range |> collect)[]
Ly = diff(y_range |> collect)[]
Lz = diff(z_range |> collect)[]
settings_axis3 = (aspect = (Lx, Ly, 5*Lz), azimuth = 0.1π, elevation = 0.2π,
                  perspectiveness=0.8, viewmode=:fitzoom,
                  xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")

# Create figure
fig = Figure(size = (1200, 900))
n = Observable(length(times))

# Create PV observable
PVₙ = @lift Array(xyzi.PV)[:,:,:,$n]

# Main 3D axis
ax = Axis3(fig[1, 1]; settings_axis3...)

surface!(ax, x_range, y_range, xyzi.bottom_height, colormap = :turbid)
volume_kwargs = (algorithm = :iso, isorange=isorange_abs, colormap=:balance, colorrange=PV_lims)
vol = volume!(ax, x_range, y_range, z_range, PVₙ, isovalue=-isovalue_abs; alpha=0.9, volume_kwargs...)
vol = volume!(ax, x_range, y_range, z_range, PVₙ, isovalue=+isovalue_abs; alpha=0.8, volume_kwargs...)

# Add colorbar
Colorbar(fig, vol, bbox=ax.scene.px_area,
         label="PV", height=25, width=Relative(0.35), vertical=false,
         alignmode = Outside(10), halign = 0.15, valign = 0.02)

# Save a snapshot as png
save("$(@__DIR__)/../figures/seamount_3d_PV_snapshot.png", fig, px_per_unit=2)

# Create title with time and parameters
title = @lift "Seamount Simulation - Roₕ = 0.2, Frₕ = 1.25;    " *
              "Time = $(@sprintf "%s" prettytime(times[$n]))"
fig[0, 1] = Label(fig, title, fontsize=18, tellwidth=false, height=8)

# Record animation
@info "Recording animation with $(length(times)) frames"
resize_to_layout!(fig)

GLMakie.record(fig, "$(@__DIR__)/../anims/3d_$(params.simname).mp4", 1:length(times),
               framerate=12, compression=30, px_per_unit=1) do frame
    @info "Frame $frame / $(length(times))"
    n[] = frame
end

@info "Animation saved successfully!"
