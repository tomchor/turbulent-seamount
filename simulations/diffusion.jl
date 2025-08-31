using LinearAlgebra
using NCDatasets: NCDataset
using ImageFiltering
using ImageFiltering.Models: solve_ROF_PD
using Printf
import Optim
using Optim: Fminbox, LBFGS

function extrude_bathymetry_3d(x, y, H, bathymetry_2d; dz=nothing, z_max=nothing)
    dx = minimum(diff(x))
    dy = minimum(diff(y))
    dz = dz isa Nothing ? (dx + dy) / 2 : dz
    zᶠ = collect(0:dz:z_max)
    zᶜ = zᶠ[1:end-1] .+ dz/2

    # Create 2D grids for x, y, z coordinates
    X = repeat(reshape(x,  :, 1, 1),         1, length(y), length(zᶠ))
    Y = repeat(reshape(y,  1, :, 1), length(x),         1, length(zᶠ))
    Z = repeat(reshape(zᶠ, 1, 1, :), length(x), length(y),          1)

    return X, Y, Z, zᶜ
end

function masked_area(smoothed, target_area; threshold=0.5)
    masked_smoothed = smoothed .> threshold
    return (sum(masked_smoothed) - target_area)^2
end

function same_area_smoothed(smoothed, unsmoothed_area; initial_guess=0.5)
    """
    Find the optimal threshold that minimizes the area difference between
    smoothed and unsmoothed area using optimization.

    Parameters:
    - smoothed: smoothed bathymetry data
    - unsmoothed_area: original unsmoothed bathymetry data

    Returns:
    - Binary mask of smoothed bathymetry with optimal threshold
    """

    # Define objective function to minimize
    objective(threshold) = abs(masked_area(smoothed, unsmoothed_area; threshold=threshold[1]))

    # Set bounds for threshold search (0 to 1 for probability-like values)
    lower_bound = [0.0]
    upper_bound = [1.0]

    # Initial guess (start with middle value)
    initial_guess = [initial_guess]

    # Use bounded optimization
    options = Optim.Options(f_abstol=0.05*unsmoothed_area, iterations=10)
    result = Optim.optimize(objective, lower_bound, upper_bound, initial_guess, Fminbox(LBFGS()), options)

    # Get optimal threshold
    optimal_threshold = Optim.minimizer(result)[1]

    if Optim.minimum(result) > 0.05*unsmoothed_area
        result = Optim.optimize(objective, lower_bound, upper_bound, 0.6.*initial_guess, Fminbox(LBFGS()), options)
        if Optim.minimum(result) > 0.05*unsmoothed_area
            result = Optim.optimize(objective, lower_bound, upper_bound, 0.2.*initial_guess, Fminbox(LBFGS()), options)
            if Optim.minimum(result) > 0.05*unsmoothed_area
                result = Optim.optimize(objective, lower_bound, upper_bound, 0.1.*initial_guess, Fminbox(LBFGS()), options)
                if Optim.minimum(result) > 0.05*unsmoothed_area
                    @warn "Optimal threshold not found"
                end
                optimal_threshold = Optim.minimizer(result)[1]
            end
            optimal_threshold = Optim.minimizer(result)[1]
        end
        optimal_threshold = Optim.minimizer(result)[1]
    end

    @info "Optimal threshold found: $optimal_threshold, objective value: $(Optim.minimum(result))"

    # Return binary mask using optimal threshold
    return smoothed .> optimal_threshold, optimal_threshold
end


function smooth_3d_bathymetry(bathymetry_3d; window_size_x, window_size_y, bc_x="circular", bc_y="replicate")
    nx, ny, nz = size(bathymetry_3d)
    kernel_x = Kernel.gaussian((window_size_x, 0))
    kernel_y = Kernel.gaussian((0, window_size_y))

    smoothed_3d = zeros(nx, ny, nz)
    initial_guess = 0.5
    for k in 1:nz
        @info "Smoothing layer $k of $nz"
        smoothed_x = imfilter(bathymetry_3d[:,:,k], kernel_x, bc_x)
        smoothed = imfilter(smoothed_x, kernel_y, bc_y)

        unsmoothed_area = sum(bathymetry_3d[:, :, k])
        smoothed_3d[:, :, k], optimal_threshold = same_area_smoothed(smoothed, unsmoothed_area; initial_guess)
        initial_guess = optimal_threshold
    end

    return smoothed_3d
end

find_interface_height(array3d::AbstractArray{Float64, 3}, x, y, z) = find_interface_height(array3d .> 0.5, x, y, z)

function find_interface_height(array3d::AbstractArray{Bool, 3}, x::AbstractVector, y::AbstractVector, z::AbstractVector)
    # Get dimensions
    nx, ny, nz = size(array3d)

    # Initialize output array
    heights = zeros(nx, ny)

    # For each x,y point, find first false value from bottom up
    for i in 1:nx
        for j in 1:ny
            # Get column of values at this x,y point
            column = view(array3d, i, j, :)

            # Find index of first false value from bottom
            k = findlast(column)

            # If found, set height to corresponding z value
            if !isnothing(k) && k > 0
                heights[i, j] = z[k]
            end
        end
    end

    return heights
end



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

# Create 2D grids for x, y, z coordinates
X, Y, Z, zᶜ = extrude_bathymetry_3d(x, y, H, elevation, z_max=1.2*H)

# Convert to Float64 and get basic info
bathymetry = Float64.(elevation)
@info "Bathymetry size: $(size(bathymetry))"
@info "Elevation range: $(extrema(bathymetry))"
@info "FWHM from data: $FWHM"

#+++ Apply three different filtering procedures

# 1. Gaussian filter using ImageFiltering.jl
@info "Applying Gaussian filter"
σ = 0.5 * FWHM / (2*dx)  # Standard deviation for Gaussian kernel
gaussian_filtered = imfilter(bathymetry, Kernel.gaussian(σ))

# # 2. Total Variation (TV) denoising using solve_ROF_PD
# @info "Applying Total Variation denoising"
# λ_tv = 10*σ^2
# max_iter_tv = 1000
# tv_filtered = solve_ROF_PD(bathymetry, λ_tv, max_iter_tv)

# 3. 3D Gaussian filter
@info "Applying 3D Gaussian filter"
bathymetry_3d = Float64.(Z .< bathymetry)
smoothed_bathymetry_3d = smooth_3d_bathymetry(bathymetry_3d; window_size_x=σ, window_size_y=σ)
gaussian_filtered_3d = find_interface_height(smoothed_bathymetry_3d, x, y, zᶜ)

#+++ Create comparison plot using GLMakie
using GLMakie

@info "Creating comparison plot"

# Create figure with 2x2 layout
fig = Figure(size = (1200, 1000));

# Define common colormap and range for all plots
elevation_range = extrema(bathymetry)
common_colormap = :terrain

# Plot original bathymetry (top left)
ax1 = Axis3(fig[1, 1], title="Original Bathymetry", xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
sf1 = surface!(ax1, x, y, bathymetry, colormap=common_colormap, colorrange=elevation_range)

# Plot Gaussian filtered (top right)
ax2 = Axis3(fig[1, 2], title="Gaussian Filtered (σ=$σ)", xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
sf2 = surface!(ax2, x, y, gaussian_filtered, colormap=common_colormap, colorrange=elevation_range)

# Plot TV filtered (bottom left)
ax3 = Axis3(fig[2, 1], title="TV Denoising (λ=$λ_tv)", xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
sf3 = surface!(ax3, x, y, tv_filtered, colormap=common_colormap, colorrange=elevation_range)

# Plot 3D Gaussian filtered (bottom right)
ax4 = Axis3(fig[2, 2], title="3D Gaussian Filtered (σ=$σ)", xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
sf4 = surface!(ax4, x, y, gaussian_filtered_3d, colormap=common_colormap, colorrange=elevation_range)

# Add a shared colorbar
Colorbar(fig[:, 3], sf1, label="Elevation [m]")

# Set z limits for all axes
for ax in (ax1, ax2, ax3, ax4)
    zlims!(ax, (0, 1.2*H))
end

# Add overall title
fig[0, 1:2] = Label(fig, "Bathymetry Filtering Comparison", fontsize=20, tellwidth=false)

# Display the figure (optional - comment out if running in batch mode)
display(fig)
#---