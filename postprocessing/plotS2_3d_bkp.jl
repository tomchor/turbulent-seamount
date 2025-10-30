using Rasters
import NCDatasets
using GLMakie
using Printf
using Oceananigans: prettytime

#+++ Preamble
fpath_xyzi_1 = "../simulations/data/xyzi.seamount_Ro_b0.1_Fr_b1_L0_FWHM500_dz1.nc"
fpath_xyzi_2 = "../simulations/data/xyzi.seamount_Ro_b0.1_Fr_b1_L0.8_FWHM500_dz1.nc"

variable = :∫⁵εₖdy

@info "Reading NetCDF files: $fpath_xyzi_1 and $fpath_xyzi_2"

# Load both datasets
xyzi_1 = RasterStack(fpath_xyzi_1, name=(variable, :bottom_height), lazy=true)
xyzi_2 = RasterStack(fpath_xyzi_2, name=(variable, :bottom_height), lazy=true)

# Get metadata and parameters from first dataset
params1 = (; (Symbol(k) => v for (k, v) in metadata(xyzi_1))...)
params2 = (; (Symbol(k) => v for (k, v) in metadata(xyzi_2))...)

# Extract grid coordinates from first dataset
xyzi_1 = xyzi_1[z_aac = 0..1.2*params1.H, x_caa = -Inf..6*params1.FWHM]
xyzi_2 = xyzi_2[z_aac = 0..1.2*params1.H, x_caa = -Inf..6*params1.FWHM]

x_range = extrema(dims(xyzi_1, :x_caa))
y_range = extrema(dims(xyzi_1, :y_aca))
z_range = extrema(dims(xyzi_1, :z_aac))
times = dims(xyzi_1[variable], :Ti)

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
settings_axis3 = (aspect = (3params1.FWHM, 3params1.FWHM, 4*Lz), azimuth = 1.6π, elevation = 0.18π,
                  perspectiveness=0.8, viewmode=:fitzoom,
                  xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")

# Create figure with two columns (3D plots and heatmaps)
fig = Figure(size = (1800, 800))

colsize!(fig.layout, 1, Relative(0.25))  # 3D plots column - 35% width
rowgap!(fig.layout, 0)
colgap!(fig.layout, 0)

# Main 3D axes - first column
ax1 = Axis3(fig[1, 1]; settings_axis3...)
ax2 = Axis3(fig[2, 1]; settings_axis3...)
for ax in (ax1, ax2)
    xlims!(ax, (-1.5params1.FWHM, 1.5params1.FWHM))
    zlims!(ax, (0, 1.1*params1.H))
end

# Heatmap axes - second column
ax1_heat = Axis(fig[1, 2], ylabel="z [m]", xticksvisible=false, xticklabelsvisible=false)
ax2_heat = Axis(fig[2, 2], ylabel="z [m]", xlabel="x [m]")

#+++ bottom height plot for 3D axes
colorrange = extrema(xyzi_1.bottom_height)
colormap = :terrain
surface!(ax1, x_range, y_range, xyzi_1.bottom_height; colormap, colorrange)
surface!(ax2, x_range, y_range, xyzi_2.bottom_height; colormap, colorrange)
#---

#+++ heatmap plots of ∫⁵εₖdy with log scale
# Extract the energy dissipation data at the final time step
εₖ_1 = xyzi_1[variable][Ti=n_final]
εₖ_2 = xyzi_2[variable][Ti=n_final]

# Create heatmaps with log scale and inferno colormap
hm1 = heatmap!(ax1_heat, x_range, z_range, εₖ_1, 
               colormap = :inferno, colorscale = log10, colorrange = (1e-7, 1e-4))
hm2 = heatmap!(ax2_heat, x_range, z_range, εₖ_2, 
               colormap = :inferno, colorscale = log10, colorrange = (1e-7, 1e-4))

# Add colorbars
Colorbar(fig[1, 3], hm1, label = "$variable [m³/s³]", height = Relative(0.8))
Colorbar(fig[2, 3], hm2, label = "$variable [m³/s³]", height = Relative(0.8))

# Add titles inside the heatmaps with white font
label_options = (space=:relative, color=:white, fontsize=16, align=(:left, :top))
text!(ax1_heat, 0.05, 0.95, text="L/FWHM = $(params1.L)"; label_options...)
text!(ax2_heat, 0.05, 0.95, text="L/FWHM = $(params2.L)"; label_options...)
#---

#+++ Create title, labels, and save the plot
title = "Roₕ = $(params1.Ro_b), Frₕ = $(params1.Fr_b); " *
        "Time = $(@sprintf "%s" prettytime(times[n_final]))"
fig[0, 1:3] = Label(fig, title, fontsize=18, tellwidth=false, height=8)

# Save the plot
save("$(@__DIR__)/../figures/seamount_3d_eps_comparison.png", fig, px_per_unit=2)
@info "Saved plot to $(@__DIR__)/../figures/seamount_3d_eps_comparison.png"
#---