using Rasters
import NCDatasets
using GLMakie
using Printf
using Oceananigans: prettytime

#+++ Preamble
fpath_xyzi_1 = "../simulations/data/xyzi.seamount_Ro_h0.1_Fr_h1.0_L0_FWHM500_dz4.nc"
fpath_xyzi_2 = "../simulations/data/xyzi.seamount_Ro_h0.1_Fr_h1.0_L0.8_FWHM500_dz4.nc"

@info "Reading NetCDF files: $fpath_xyzi_1 and $fpath_xyzi_2"

# Load both datasets
xyzi_1 = RasterStack(fpath_xyzi_1, name=(:∫⁵εₖdx, :bottom_height), lazy=true)
xyzi_2 = RasterStack(fpath_xyzi_2, name=(:∫⁵εₖdx, :bottom_height), lazy=true)

# Get metadata and parameters from first dataset
md = metadata(xyzi_1)
params = (; (Symbol(k) => v for (k, v) in md)...)

# Extract grid coordinates from first dataset
xyzi_1 = xyzi_1[z_aac = 0..1.1*params.H, x_caa = -Inf..6*params.FWHM]
xyzi_2 = xyzi_2[z_aac = 0..1.1*params.H, x_caa = -Inf..6*params.FWHM]

x_range = extrema(dims(xyzi_1, :x_caa))
y_range = extrema(dims(xyzi_1, :y_aca))
z_range = extrema(dims(xyzi_1, :z_aac))
times = dims(xyzi_1.∫⁵εₖdx, :Ti)

# Use the last time step for static plotting
n_final = length(times)
#---

#+++ Set limits based on data range or physical considerations
isovalue_εₚ₁ = 5e-10
isovalue_εₚ₂ = 1e-9
εₚ_range = (isovalue_εₚ₁, isovalue_εₚ₂)
isorange_εₚ₁ = isovalue_εₚ₁/2
isorange_εₚ₂ = isovalue_εₚ₂/2
#---

# 3D plot settings
Lx = diff(x_range |> collect)[]
Ly = diff(y_range |> collect)[]
Lz = diff(z_range |> collect)[]
settings_axis3 = (aspect = (3params.FWHM, 3params.FWHM, 4*Lz), azimuth = 1.6π, elevation = 0.18π,
                  perspectiveness=0.8, viewmode=:fitzoom,
                  xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")

# Create figure with two columns (3D plots and heatmaps)
fig = Figure(size = (1800, 800))

# Main 3D axes - first column
ax1 = Axis3(fig[1, 1]; settings_axis3...)
ax2 = Axis3(fig[2, 1]; settings_axis3...)
xlims!(ax1, (-1.5params.FWHM, 1.5params.FWHM))
xlims!(ax2, (-1.5params.FWHM, 1.5params.FWHM))

# Heatmap axes - second column
ax1_heat = Axis(fig[1, 2], xlabel="x [m]", ylabel="z [m]", title="L = $(params.L) m")
ax2_heat = Axis(fig[2, 2], xlabel="x [m]", ylabel="z [m]", title="L = $(params.L) m")


#+++ bottom height plot for 3D axes
surface!(ax1, x_range, y_range, xyzi_1.bottom_height, colormap = :turbid)
surface!(ax2, x_range, y_range, xyzi_2.bottom_height, colormap = :turbid)
#---

#+++ heatmap plots of ∫⁵εₖdx with log scale
# Extract the energy dissipation data at the final time step
εₖ_1 = xyzi_1.∫⁵εₖdx[Ti=n_final]
εₖ_2 = xyzi_2.∫⁵εₖdx[Ti=n_final]

# Create heatmaps with log scale and inferno colormap
hm1 = heatmap!(ax1_heat, x_range, z_range, εₖ_1, 
               colormap = :inferno, colorscale = log10, colorrange = (1e-5, 1e-3))
hm2 = heatmap!(ax2_heat, x_range, z_range, εₖ_2, 
               colormap = :inferno, colorscale = log10, colorrange = (1e-5, 1e-3))

# Add colorbars
Colorbar(fig[1, 3], hm1, label = "∫⁵εₖdx [m³/s³]", height = Relative(0.8))
Colorbar(fig[2, 3], hm2, label = "∫⁵εₖdx [m³/s³]", height = Relative(0.8))
#---

#+++ Create title, labels, and save the plot
title = "Roₕ = $(params.Ro_h), Frₕ = $(params.Fr_h), L = $(params.L) m; " *
        "Time = $(@sprintf "%s" prettytime(times[n_final]))"
fig[0, 1:3] = Label(fig, title, fontsize=18, tellwidth=false, height=8)

# Save the plot
save("$(@__DIR__)/../figures/seamount_3d_eps_comparison.png", fig, px_per_unit=2)
#---