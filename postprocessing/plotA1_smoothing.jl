using LinearAlgebra
using NCDatasets: NCDataset
using Printf

include("../simulations/utils.jl")

#+++ Read bathymetry data
@info "Reading bathymetry data"
bathymetry_filepath = joinpath(@__DIR__, "../bathymetry/balanus-GMRT-bathymetry-preprocessed.nc")
ds_bathymetry = NCDataset(bathymetry_filepath)
x = ds_bathymetry["x"] |> collect
y = ds_bathymetry["y"] |> collect

dx = minimum(diff(x))
H = ds_bathymetry.attrib["H"]

FWHM = ds_bathymetry.attrib["FWHM"]
elevation = ds_bathymetry["periodic_elevation"] |> collect

# Close the original dataset
close(ds_bathymetry)

# Convert to Float64 and get basic info
@info "Bathymetry size: $(size(elevation))"
@info "Elevation range: $(extrema(elevation))"
@info "FWHM from data: $FWHM"

#+++ Apply filtering procedures with different sigma values

# Define sigma values as fractions of FWHM
sigma_fractions = [0.05, 0.1, 0.4, 0.8]
sigma_values = sigma_fractions .* FWHM

@info "Applying filters for different sigma values: $sigma_values"

# Store filtered results
gaussian_results = []
gaussian_3d_results = []

for σ in sigma_values
    @info "Processing σ = $σ"

    # Gaussian filter
    gaussian_filtered = smooth_bathymetry(elevation, x, y; scale_x=σ, scale_y=σ)
    push!(gaussian_results, gaussian_filtered)

    # 3D Gaussian filter
    gaussian_filtered_3d = smooth_bathymetry_3d(elevation, x, y; scale_x=σ, scale_y=σ, dz=dx/6, bathymetry_filepath)
    push!(gaussian_3d_results, gaussian_filtered_3d)
end

#+++ Create comparison plot using GLMakie
using GLMakie

@info "Creating comparison plot"

# Create figure with 2 rows x 6 columns (original + 5 sigma values)
fig = Figure(size = (2400, 800));

# Define common colormap and range for all plots
elevation_range = extrema(elevation)
common_colormap = :terrain

# Plot original elevation (leftmost column)
ax_orig_1 = Axis3(fig[1, 1], title="Original Bathymetry",
                  xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
sf_orig = surface!(ax_orig_1, x, y, elevation, colormap=common_colormap, colorrange=elevation_range)
zlims!(ax_orig_1, (0, 1.2*H))

ax_orig_2 = Axis3(fig[2, 1], title="Original Bathymetry",
                  xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
surface!(ax_orig_2, x, y, elevation, colormap=common_colormap, colorrange=elevation_range)
zlims!(ax_orig_2, (0, 1.2*H))

# Plot Gaussian filtered results (top row, columns 2-6)
for (i, (σ, result)) in enumerate(zip(sigma_values, gaussian_results))
    ax = Axis3(fig[1, i+1], title="Gaussian (σ=$(round(σ/FWHM, digits=2))×FWHM)",
               xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
    surface!(ax, x, y, result, colormap=common_colormap, colorrange=elevation_range)
    zlims!(ax, (0, 1.2*H))
end

# Plot 3D Gaussian filtered results (bottom row, columns 2-6)
for (i, (σ, result)) in enumerate(zip(sigma_values, gaussian_3d_results))
    ax = Axis3(fig[2, i+1], title="3D Gaussian (σ=$(round(σ/FWHM, digits=2))×FWHM)",
               xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
    surface!(ax, x, y, result, colormap=common_colormap, colorrange=elevation_range)
    zlims!(ax, (0, 1.2*H))
end

# Add row labels
Label(fig[1, 0], "Gaussian Filter", rotation=π/2, tellheight=false, fontsize=16)
Label(fig[2, 0], "3D Gaussian Filter", rotation=π/2, tellheight=false, fontsize=16)

# Add a shared colorbar
ncols = length(sigma_values) + 2
Colorbar(fig[:, ncols], sf_orig, label="Elevation [m]")

# Add overall title
fig[0, 1:ncols] = Label(fig, "Bathymetry Filtering Comparison - Multiple Sigma Values", fontsize=20, tellwidth=false)

# Display the figure (optional - comment out if running in batch mode)
display(fig)
#---
