using LinearAlgebra
using NCDatasets: NCDataset
using ImageFiltering
using ImageFiltering.Models: solve_ROF_PD
using Printf

function anisotropic_diffusion_2d(image; K=10, num_iter=50, lambda=0.25)
    """
    Apply Perona-Malik anisotropic diffusion to 2D data
    
    Parameters:
    - image: 2D array to smooth
    - K: gradient threshold parameter
    - num_iter: number of iterations
    - lambda: time step (should be ≤ 0.25 for stability)
    """
    
    u = copy(Float64.(image))
    rows, cols = size(u)
    
    for iter in 1:num_iter
        # Compute gradients in all directions
        grad_N = zeros(rows, cols)  # North
        grad_S = zeros(rows, cols)  # South  
        grad_E = zeros(rows, cols)  # East
        grad_W = zeros(rows, cols)  # West
        
        # Northern gradient
        grad_N[2:end, :] = u[1:end-1, :] - u[2:end, :]
        
        # Southern gradient  
        grad_S[1:end-1, :] = u[2:end, :] - u[1:end-1, :]
        
        # Eastern gradient
        grad_E[:, 1:end-1] = u[:, 2:end] - u[:, 1:end-1]
        
        # Western gradient
        grad_W[:, 2:end] = u[:, 1:end-1] - u[:, 2:end]
        
        # Compute diffusion coefficients using exponential function
        # c(∇u) = exp(-(|∇u|/K)²)
        c_N = exp.(-(grad_N ./ K).^2)
        c_S = exp.(-(grad_S ./ K).^2)  
        c_E = exp.(-(grad_E ./ K).^2)
        c_W = exp.(-(grad_W ./ K).^2)
        
        # Apply anisotropic diffusion update
        u += lambda * (c_N .* grad_N + c_S .* grad_S + 
                      c_E .* grad_E + c_W .* grad_W)
    end
    
    return u
end

# Alternative diffusion coefficient functions
function diffusion_coeff_exponential(grad_mag, K)
    return exp.(-(grad_mag ./ K).^2)
end

function diffusion_coeff_rational(grad_mag, K)
    return 1 ./ (1 .+ (grad_mag ./ K).^2)
end

function diffusion_coeff_tukey(grad_mag, K)
    mask = grad_mag .<= K
    result = zeros(size(grad_mag))
    result[mask] = (1 .- (grad_mag[mask] ./ K).^2).^2
    return result
end

function extrude_bathymetry_3d(x, y, H, bathymetry_2d; dz=nothing, z_max=nothing)
    dx = minimum(diff(x))
    dy = minimum(diff(y))
    dz = dz isa Nothing ? (dx + dy) / 2 : dz
    zᶠ = collect(0:dz:z_max)
    zᶜ = zᶠ[1:end-1] + dz/2

    # Create 2D grids for x, y, z coordinates
    X = repeat(reshape(x,  :, 1, 1),         1, length(y), length(zᶠ))
    Y = repeat(reshape(y,  1, :, 1), length(x),         1, length(zᶠ))
    Z = repeat(reshape(zᶠ, 1, 1, :), length(x), length(y),          1)

    return X, Y, Z, zᶜ
end

function smooth_3d_bathymetry(bathymetry_3d; window_size_x, window_size_y, bc_x="circular", bc_y="replicate")
    kernel_x = Kernel.gaussian((window_size_x, 0, 0))
    kernel_y = Kernel.gaussian((0, window_size_y, 0))

    smoothed_x = imfilter(bathymetry_3d, kernel_x, bc_x)
    smoothed = imfilter(smoothed_x, kernel_y, bc_y)

    return smoothed
end

function find_interface_height(array3d::AbstractArray{Bool,3}, x::AbstractVector, y::AbstractVector, z::AbstractVector)
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
                heights[i,j] = z[k]
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
X, Y, Z = extrude_bathymetry_3d(x, y, H, elevation, z_max=1.2*H)

# Convert to Float64 and get basic info
bathymetry = Float64.(elevation)
@info "Bathymetry size: $(size(bathymetry))"
@info "Elevation range: $(extrema(bathymetry))"
@info "FWHM from data: $FWHM"

bathymetry_3d = Float64.(Z .< bathymetry)
smoothed_bathymetry_3d = smooth_3d_bathymetry(bathymetry_3d; window_size_x=σ, window_size_y=σ)

#+++ Apply three different filtering procedures

# 1. Gaussian filter using ImageFiltering.jl
@info "Applying Gaussian filter"
σ = 0.8 * FWHM / (2*dx)  # Standard deviation for Gaussian kernel
gaussian_filtered = imfilter(bathymetry, Kernel.gaussian(σ))

# 2. Total Variation (TV) denoising using solve_ROF_PD
@info "Applying Total Variation denoising"
λ_tv = 0.1*σ^2
max_iter_tv = 1000
tv_filtered = solve_ROF_PD(bathymetry, λ_tv, max_iter_tv)

# 3. Anisotropic diffusion using the custom function
@info "Applying anisotropic diffusion"
K_aniso = σ^2  # Gradient threshold parameter
num_iter_aniso = 50
λ_aniso = 0.2  # Time step
aniso_filtered = anisotropic_diffusion_2d(bathymetry; K=K_aniso, num_iter=num_iter_aniso, lambda=λ_aniso)


#+++ Create comparison plot using GLMakie
using GLMakie

@info "Creating comparison plot"

# Create figure with 2x2 layout
fig = Figure(size = (1200, 1000))

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

# Plot anisotropic diffusion filtered (bottom right)
ax4 = Axis3(fig[2, 2], title="Anisotropic Diffusion (K=$K_aniso)", xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
sf4 = surface!(ax4, x, y, aniso_filtered, colormap=common_colormap, colorrange=elevation_range)

# Add a shared colorbar
Colorbar(fig[:, 3], sf1, label="Elevation [m]")

# Add overall title
fig[0, 1:2] = Label(fig, "Bathymetry Filtering Comparison", fontsize=20, tellwidth=false)

# Display the figure (optional - comment out if running in batch mode)
display(fig)