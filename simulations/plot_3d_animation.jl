using Rasters
import NCDatasets
using GLMakie
using Printf
using Oceananigans: prettytime

# Data file path
fpath_xyzi = (@isdefined simulation) ? simulation.output_writers[:nc_xyzi].filepath : "data/xyzi.seamount_Ro_h=0.2_Fr_h=1.25_L=0_dz=4_closure=DSM.nc"

@info "Reading NetCDF file: $fpath_xyzi"
xyzi = RasterStack(fpath_xyzi, name=(:PV, :εₖ, :bottom_height), lazy=true)

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
interior_q = params.N²∞ * params.f₀
isovalue_q = 1.4 * interior_q  # Adjust this based on your data
isorange_q = isovalue_q/10
PV_range = 1.2 .* (-isovalue_q, +isovalue_q)

isovalue_ε1 = 5e-9
isovalue_ε2 = 1e-7
ε_range = (isovalue_ε1, isovalue_ε2)
isorange_ε1 = isovalue_ε1/2
isorange_ε2 = isovalue_ε2/3

# 3D plot settings
Lx = diff(x_range |> collect)[]
Ly = diff(y_range |> collect)[]
Lz = diff(z_range |> collect)[]
settings_axis3 = (aspect = (Lx, Ly, 5*Lz), azimuth = 0.6π, elevation = 0.18π,
                  perspectiveness=0.8, viewmode=:fitzoom,
                  xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")

# Create figure
fig = Figure(size = (1300, 600))
n = Observable(length(times))

# Create PV observable
PVₙ = @lift Array(xyzi.PV)[:,:,:,$n]
εₙ = @lift Array(xyzi.εₖ)[:,:,:,$n]

# Main 3D axis
ax1 = Axis3(fig[1, 1]; settings_axis3...)
ax2 = Axis3(fig[1, 2]; settings_axis3...)

#region bottom height plot
for ax in (ax1, ax2)
    surface!(ax, x_range, y_range, xyzi.bottom_height, colormap = :turbid)
end
#endregion

#region PV plot
vol1_kwargs = (algorithm = :iso, isorange=isorange_q, colormap=:balance, colorrange=PV_range)
vol1 = volume!(ax1, x_range, y_range, z_range, PVₙ, isovalue=-isovalue_q; alpha=0.9, vol1_kwargs...)
vol1 = volume!(ax1, x_range, y_range, z_range, PVₙ, isovalue=+isovalue_q; alpha=0.7, vol1_kwargs...)
Colorbar(fig, vol1, bbox=ax1.scene.viewport,
         label="PV", height=15, width=Relative(0.5), vertical=false,
         alignmode = Outside(10), halign = 0.15, valign = 1.02)
#endregion

#region εₖ plot
vol2_kwargs1 = (algorithm = :iso, colormap=Reverse(:roma), colorrange=ε_range)
vol2_kwargs2 = (algorithm = :iso, colormap=Reverse(:roma), colorrange=ε_range)
#vol2 = volume!(ax2, x_range, y_range, z_range, εₙ, isovalue=isovalue_ε1, isorange=isorange_ε1, alpha=0.5; vol2_kwargs1...)
vol2 = volume!(ax2, x_range, y_range, z_range, εₙ, isovalue=isovalue_ε2, isorange=isorange_ε2, alpha=0.9; vol2_kwargs2...)
Colorbar(fig, vol2, bbox=ax2.scene.viewport,
         label="εₖ", height=15, width=Relative(0.5), vertical=false,
         alignmode = Outside(10), halign = 0.75, valign = 1.02)
#endregion

# Save a snapshot as png
save("$(@__DIR__)/../figures/seamount_3d_PV_snapshot.png", fig, px_per_unit=2)

# Create title with time and parameters
title = @lift "Roₕ = $(params.Ro_h), Frₕ = $(params.Fr_h), L = $(params.L) m, dz = $(params.dz) m;    " *
              "Time = $(@sprintf "%s" prettytime(times[$n]))"
fig[0, 1:2] = Label(fig, title, fontsize=18, tellwidth=false, height=8)

# Record animation
@info "Recording animation with $(length(times)) frames"
resize_to_layout!(fig)

GLMakie.record(fig, "$(@__DIR__)/../anims/3d_$(params.simname).mp4", 1:length(times),
               framerate=14, compression=30, px_per_unit=2) do frame
    @info "Frame $frame / $(length(times))"
    n[] = frame
end

@info "Animation saved successfully!"
