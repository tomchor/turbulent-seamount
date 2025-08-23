using Rasters
import NCDatasets
using GLMakie
using Printf
using Oceananigans: prettytime

#+++ Preamble
fpath_xyzi_1 = "../simulations/data/xyzi.seamount_Ro_h0.2_Fr_h1.25_L0_FWHM400_dz4.nc"
fpath_xyzi_2 = "../simulations/data/xyzi.seamount_Ro_h0.2_Fr_h1.25_L0.8_FWHM400_dz4.nc"

@info "Reading NetCDF files: $fpath_xyzi_1 and $fpath_xyzi_2"

# Load both datasets
xyzi_1 = RasterStack(fpath_xyzi_1, name=(:εₚ, :bottom_height), lazy=true)
xyzi_2 = RasterStack(fpath_xyzi_2, name=(:εₚ, :bottom_height), lazy=true)

# Get metadata and parameters from first dataset
md = metadata(xyzi_1)
params = (; (Symbol(k) => v for (k, v) in md)...)

# Extract grid coordinates from first dataset
xyzi_1 = xyzi_1[z_aac = 0..1.1*params.H, y_aca = -Inf..5.5*params.FWHM]
xyzi_2 = xyzi_2[z_aac = 0..1.1*params.H, y_aca = -Inf..5.5*params.FWHM]

x_range = extrema(dims(xyzi_1, :x_caa))
y_range = extrema(dims(xyzi_1, :y_aca))
z_range = extrema(dims(xyzi_1, :z_aac))
times = dims(xyzi_1.εₚ, :Ti)

# Use the last time step for static plotting
n_final = length(times)
#---

#+++ Set limits based on data range or physical considerations
isovalue_εₚ₁ = 3e-10
isovalue_εₚ₂ = 1e-9
εₚ_range = (isovalue_εₚ₁, isovalue_εₚ₂)
isorange_εₚ₁ = isovalue_εₚ₁/2
isorange_εₚ₂ = isovalue_εₚ₂/2
#---

# 3D plot settings
Lx = diff(x_range |> collect)[]
Ly = diff(y_range |> collect)[]
Lz = diff(z_range |> collect)[]
settings_axis3 = (aspect = (Lx, Ly, 5*Lz), azimuth = 1.6π, elevation = 0.18π,
                  perspectiveness=0.8, viewmode=:fitzoom,
                  xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")

# Create figure with two columns
fig = Figure(size = (1950, 600))

# Create observables for final time step
εₚₙ_1 = @lift Array(xyzi_1.εₚ)[:,:,:,$n_final]
εₚₙ_2 = @lift Array(xyzi_2.εₚ)[:,:,:,$n_final]

# Main 3D axes - two columns
ax1_εₚ = Axis3(fig[1, 1]; settings_axis3...)
ax2_εₚ = Axis3(fig[1, 2]; settings_axis3...)

#+++ bottom height plot for all axes
surface!(ax1_εₚ, x_range, y_range, xyzi_1.bottom_height, colormap = :turbid)
surface!(ax2_εₚ, x_range, y_range, xyzi_2.bottom_height, colormap = :turbid)
#---

#+++ PV plots (first column)
vol1_kwargs = (algorithm = :iso, colormap=:balance, transparency = true, isorange=isorange_εₚ₁, colorrange=εₚ_range)
vol1_1 = volume!(ax1_εₚ, x_range, y_range, z_range, εₚₙ_1, isovalue=isovalue_εₚ₁, isorange=isorange_εₚ₁, alpha=0.7; vol1_kwargs...)
vol1_1 = volume!(ax1_εₚ, x_range, y_range, z_range, εₚₙ_1, isovalue=isovalue_εₚ₂, isorange=isorange_εₚ₂, alpha=0.9; vol1_kwargs...)

# PV plots (second column)
vol1_2 = volume!(ax2_εₚ, x_range, y_range, z_range, εₚₙ_2, isovalue=isovalue_εₚ₁, isorange=isorange_εₚ₁, alpha=0.7; vol1_kwargs...)
vol1_2 = volume!(ax2_εₚ, x_range, y_range, z_range, εₚₙ_2, isovalue=isovalue_εₚ₂, isorange=isorange_εₚ₂, alpha=0.9; vol1_kwargs...)

# Colorbars for εₚ
Colorbar(fig, vol1_1, bbox=ax1_εₚ.scene.viewport,
         label="εₚ", height=15, width=Relative(0.5), vertical=false,
         alignmode = Outside(10), halign = 0.15, valign = 1.02)

Colorbar(fig, vol1_2, bbox=ax2_εₚ.scene.viewport,
         label="εₚ", height=15, width=Relative(0.5), vertical=false,
         alignmode = Outside(10), halign = 0.15, valign = 1.02)
#---

#+++ Create title, labels, and save the plot
title = "Roₕ = $(params.Ro_h), Frₕ = $(params.Fr_h), L = $(params.L) m; " *
        "Time = $(@sprintf "%s" prettytime(times[n_final]))"
fig[0, 1:2] = Label(fig, title, fontsize=18, tellwidth=false, height=8)

# Add column labels
fig[0, 1] = Label(fig, "Dataset 1 (L=$(metadata(xyzi_1)["L"]) FWHM)", fontsize=14, tellwidth=false, height=6)
fig[0, 2] = Label(fig, "Dataset 2 (L=$(metadata(xyzi_2)["L"]) FWHM)", fontsize=14, tellwidth=false, height=6)

# Save the plot
save("$(@__DIR__)/../figures/seamount_3d_eps_p_comparison.png", fig, px_per_unit=2)
#---