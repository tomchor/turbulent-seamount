using LinearAlgebra
using NCDatasets: NCDataset
using Printf


include("utils.jl")



#+++ Read bathymetry data
@info "Reading bathymetry data"
ds_bathymetry = NCDataset(joinpath(@__DIR__, "../bathymetry/balanus-bathymetry-preprocessed.nc"))
x = ds_bathymetry["x"] |> collect
y = ds_bathymetry["y"] |> collect

dx = minimum(diff(x))
dy = minimum(diff(y))
H = ds_bathymetry.attrib["H"]

FWHM = 2 * ds_bathymetry.attrib["FWHM"]
elevation = ds_bathymetry["periodic_elevation"] |> collect

# Close the original dataset
close(ds_bathymetry)

# Convert to Float64 and get basic info
@info "Bathymetry size: $(size(elevation))"
@info "Elevation range: $(extrema(elevation))"
@info "FWHM from data: $FWHM"

#+++ Apply three different filtering procedures

# 1. Gaussian filter using ImageFiltering.jl
@info "Applying Gaussian filter"
σ = 0.3 * FWHM
gaussian_filtered = smooth_bathymetry(elevation, x, y; scale_x=σ, scale_y=σ)

# 2. 3D Gaussian filter
@info "Applying 3D Gaussian filter"
gaussian_filtered_3d = smooth_bathymetry_3d(elevation, x, y; scale_x=σ, scale_y=σ)

#+++ Create comparison plot using GLMakie
using GLMakie

@info "Creating comparison plot"

# Create figure with 2x2 layout
fig = Figure(size = (1200, 1000));

# Define common colormap and range for all plots
elevation_range = extrema(elevation)
common_colormap = :terrain

# Plot original bathymetry (top left)
ax1 = Axis3(fig[1, 1], title="Original Bathymetry", xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
sf1 = surface!(ax1, x, y, elevation, colormap=common_colormap, colorrange=elevation_range)

# Plot Gaussian filtered (top right)
ax2 = Axis3(fig[1, 2], title="Gaussian Filtered (σ=$σ)", xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
sf2 = surface!(ax2, x, y, gaussian_filtered, colormap=common_colormap, colorrange=elevation_range)

# Plot 3D Gaussian filtered (bottom right)
ax4 = Axis3(fig[2, 2], title="3D Gaussian Filtered (σ=$σ)", xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
sf4 = surface!(ax4, x, y, gaussian_filtered_3d, colormap=common_colormap, colorrange=elevation_range)

# Add a shared colorbar
Colorbar(fig[:, 3], sf1, label="Elevation [m]")

# Set z limits for all axes
for ax in (ax1, ax2, ax4)
    zlims!(ax, (0, 1.2*H))
end

# Add overall title
fig[0, 1:2] = Label(fig, "Bathymetry Filtering Comparison", fontsize=20, tellwidth=false)

# Display the figure (optional - comment out if running in batch mode)
display(fig)
#---