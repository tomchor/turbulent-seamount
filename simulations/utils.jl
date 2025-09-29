using Parameters
using ImageFiltering: imfilter, Kernel
import Optim
using Optim: GoldenSection, Fminbox, LBFGS, optimize
using CUDA: devices, device!, functional, totalmem, name, available_memory, memory_status, @time
using NCDatasets: NCDataset, defDim, defVar
using Dates: now

import Interpolations # To use Flat in a way that doesn't conflict with Oceananigans.Flat
using Interpolations: interpolate, BSpline, extrapolate, OnGrid, Cubic, Line
using Statistics: mean

#+++ Get good grid size
""" Rounds `a` to the nearest even number """
nearest_even(a) = Int(2*round(round(a) / 2))

""" Returns Nx Ny Nz based on total number of points `N`"""
function get_sizes(N; Lx=8, Ly=8, Lz=0.5, aspect_ratio_x=3.2, aspect_ratio_y=3.2)
    Cz = Lx * Ly / (Lz^2 * aspect_ratio_x * aspect_ratio_y)
    Nz = (N / Cz)^(1/3)
    Nx = Lx * Nz / (Lz * aspect_ratio_x)
    Ny = Ly * Nz / (Lz * aspect_ratio_y)
    Nx = round(Int, Nx)
    Ny = round(Int, Ny)
    Nz = round(Int, max(2, Nz))
    return (; Nx, Ny, Nz)
end

# Function to generate multiples of prime factors up to a limit
function closest_factor_number(primes::NTuple{3, Int}, target::Int)
    closest_number = 1
    min_difference = abs(target - closest_number)
    # We will iterate over different combinations of powers of primes
    for i in 0:15  # You can adjust this loop depth
        for j in 0:15
            for k in 0:15
                # Generate the product of primes with different powers
                product = primes[1]^i * primes[2]^j * primes[3]^k
                diff = abs(target - product)
                if diff < min_difference
                    min_difference = diff
                    closest_number = product
                end
            end
        end
    end
    return closest_number
end

function closest_factor_number(primes::NTuple{2, Int}, target::Int)
    closest_number = 1
    min_difference = abs(target - closest_number)
    # We will iterate over different combinations of powers of primes
    for i in 0:15 # You can adjust this loop depth
        for j in 0:15
            # Generate the product of primes with different powers
            product = primes[1]^i * primes[2]^j
            diff = abs(target - product)
            if diff < min_difference
                min_difference = diff
                closest_number = product
            end
        end
    end
    return closest_number
end

function closest_factor_number(primes::NTuple{1, Int}, target::Int)
    closest_number = 1
    min_difference = abs(target - closest_number)
    # We will iterate over different combinations of powers of primes
    for i in 0:15 # You can adjust this loop depth
        # Generate the product of primes with different powers
        product = primes[1]^i
        diff = abs(target - product)
        if diff < min_difference
            min_difference = diff
            closest_number = product
        end
    end
    return closest_number
end
#---

#+++ Smooth bathymetry using regular 2D Gaussian filter
function smooth_bathymetry_with_gaussian(elevation; window_size_x, window_size_y, bc_x="circular", bc_y="replicate", target_height=nothing)
    # Create separate Gaussian kernels for x and y directions
    kernel_x = Kernel.gaussian((window_size_x, 0))
    kernel_y = Kernel.gaussian((0, window_size_y))

    # Apply filters sequentially with different boundary conditions
    # First smooth in x direction
    smoothed_x = imfilter(elevation, kernel_x, bc_x)
    # Then smooth in y direction
    smoothed = imfilter(smoothed_x, kernel_y, bc_y)

    # Rescale height if target_height is specified
    if target_height !== nothing
        current_max_height = maximum(smoothed)
        if current_max_height > 0
            scaling_factor = target_height / current_max_height
            smoothed = smoothed .* scaling_factor
        end
    end

    return smoothed
end

function smooth_bathymetry_with_gaussian(elevation, x, y; scale_x, scale_y, kwargs...)
    Δx_min = minimum(diff(x))
    Δy_min = minimum(diff(y))

    # The factor of 2 is there because the Gaussian kernel's standard deviation
    # should be about half the desired smoothing length
    window_size_x = scale_x / (2 * Δx_min)
    window_size_y = scale_y / (2 * Δy_min)

    # Call the original method with calculated window sizes
    return smooth_bathymetry_with_gaussian(elevation; window_size_x, window_size_y, kwargs...)
end
#---

#+++ Smooth bathymetry by coarsening and then interpolating
"""
    coarsen_bathymetry(elevation, x, y; window_x=2, window_y=2)

Coarsens the grid using window averaging.

Returns coarse x and y, and elevation array.
"""
function coarsen_bathymetry(elevation, x, y; window_x=2, window_y=2)
    nx, ny = size(elevation)
    dx = minimum(diff(x))
    dy = minimum(diff(y))

    # Create coordinate vectors (assuming unit spacing)
    x_coarse = x[1]:window_x*dx:x[end] |> collect
    y_coarse = y[1]:window_y*dy:y[end] |> collect

    # Calculate coarse grid dimensions
    nx_coarse = cld(nx, window_x) # ceiling division
    ny_coarse = cld(ny, window_y) # ceiling division

    # Initialize coarse elevation array
    elevation_coarse = zeros(Float64, nx_coarse, ny_coarse)

    # Compute coarse values as mean of high-res values in each window
    for i in 1:nx_coarse, j in 1:ny_coarse
            # Define window bounds
            x_coarse_start = 1 + (i-1) * window_x
            x_coarse_end = min(i * window_x, nx)
            y_coarse_start = 1 + (j-1) * window_y
            y_coarse_end = min(j * window_y, ny)

            # Extract window data
            window_data = elevation[x_coarse_start:x_coarse_end, y_coarse_start:y_coarse_end]

            # Compute mean, handling missing values
            if any(ismissing.(window_data))
                # Filter out missing values and compute mean of non-missing
                valid_data = filter(!ismissing, window_data)
                elevation_coarse[i, j] = isempty(valid_data) ? 0.0 : mean(valid_data)
            else
                elevation_coarse[i, j] = mean(window_data)
            end
    end

    return x_coarse, y_coarse, elevation_coarse
end

"""
    smooth_bathymetry_with_coarsening(elevation, x, y; scale_x, scale_y, clip_negative=true)

Smooths bathymetry by coarsening the grid using window averaging, and then upscaling again by interpolating.

# Arguments
- `elevation`: 2D array of elevation values
- `x`: 1D array of x coordinates
- `y`: 1D array of y coordinates
- `scale_x`: Scale factor for x direction
- `scale_y`: Scale factor for y direction
- `clip_negative`: If true, clips negative values to zero before returning (default: true)

# Returns
- Smoothed elevation array with same dimensions (and coordinates) as input
"""
function smooth_bathymetry_with_coarsening(elevation, x, y; scale_x, scale_y, clip_negative=true)
    dx = minimum(diff(x))
    dy = minimum(diff(y))

    window_x = scale_x / (2 * dx) |> round |> Int
    window_y = scale_y / (2 * dy) |> round |> Int

    x_coarse, y_coarse, elevation_coarse = coarsen_bathymetry(elevation, x, y; window_x, window_y)
    nx_coarse, ny_coarse = map(length, (x_coarse, y_coarse))

    # In order to make bicubic interpolation, we need to assume that the grid is uniform
    # and then normalize the coordinates to the range of the coarse grid
    # and then interpolate. This is a hack to get around the fact that the grid is not uniform and
    # Interpolations.jl does not support bicubic interpolation on non-uniform grids.
    itp = interpolate(elevation_coarse, BSpline(Cubic(Line(OnGrid()))), OnGrid())
    itp = extrapolate(itp, Interpolations.Flat())
    x_norm = (x .- x[1]) / (x[end] - x[1]) * nx_coarse # From 0 to nx_coarse
    y_norm = (y .- y[1]) / (y[end] - y[1]) * ny_coarse # From 0 to ny_coarse

    elevation_smooth = itp.(reshape(x_norm, (length(x_norm), 1)), reshape(y_norm, (1, length(y_norm))))

    # Clip negative values to zero if requested
    clip_negative && (elevation_smooth = max.(elevation_smooth, 0.0))

    return elevation_smooth
end
#---

#+++ Functions to create z coordinate
function create_stretched_z_coordinates(dz, H, Lz, stretching_factor = 1.1, min_stretching_factor = 1.5)
    # Uniform spacing from 0 to H
    z_uniform = 0:dz:H

    # Stretched spacing from H to Lz
    z_stretched = [H]
    current_z = H
    current_dz = dz

    while current_z < Lz
        current_dz *= stretching_factor
        next_z = current_z + current_dz

        # Check if the remaining distance is too small compared to current spacing
        remaining_distance = Lz - current_z
        if remaining_distance < current_dz * min_stretching_factor
            # Adjust the final point to have reasonable spacing
            final_spacing = remaining_distance
            push!(z_stretched, Lz)
            break
        else
            current_z = next_z
            if current_z <= Lz
                push!(z_stretched, current_z)
            else
                # If we would overshoot, place the final point at Lz
                push!(z_stretched, Lz)
                break
            end
        end
    end

    # Combine the arrays, removing the duplicate H point
    z_coords = vcat(collect(z_uniform), z_stretched[2:end])

    return z_coords
end

# Function to count grid points for a given stretching factor
function count_grid_points(stretching_factor, dz, H, Lz)
    z_coords_test = create_stretched_z_coordinates(dz, H, Lz, stretching_factor)
    return length(z_coords_test) - 1  # Number of grid cells
end

"""
    create_optimal_z_coordinates(dz, H, Lz, prime_factors;
                                 initial_stretching_factor=1.1,
                                 min_stretching_factor=1.5,
                                 search_bounds=(1.05, 1.5),
                                 verbose=true)

Create z-coordinates with uniform spacing from 0 to H, then stretched spacing to Lz,
ensuring the total number of grid cells (Nz) is a product of the given prime factors.

# Arguments
- `dz`: Grid spacing for the uniform region [0, H]
- `H`: Height up to which uniform spacing is used
- `Lz`: Total domain height
- `prime_factors`: Tuple of prime factors (e.g., (2, 3, 5))
- `initial_stretching_factor`: Initial guess for stretching factor
- `min_stretching_factor`: Minimum ratio for final spacing adjustment
- `search_bounds`: Tuple of (lower, upper) bounds for optimization
- `verbose`: Whether to print information messages

# Returns
- `z_coords`: Vector of z-coordinates from 0 to Lz
"""
function create_optimal_z_coordinates(dz, H, Lz, prime_factors;
                                      initial_stretching_factor = 1.1,
                                      min_stretching_factor = 1.5,
                                      search_bounds = (1.05, 1.5),
                                      verbose = true)

    # Create initial stretched z-coordinates
    z_coords = create_stretched_z_coordinates(dz, H, Lz, initial_stretching_factor, min_stretching_factor)

    if verbose
        @info "Created initial stretched z-grid with $(length(z_coords)) points"
        @info "Uniform spacing from 0 to $H, then stretched to $Lz"
    end

    # Find optimal Nz that's a factor of the given primes
    initial_Nz = length(z_coords) - 1  # Number of grid cells
    optimal_Nz = closest_factor_number(prime_factors, initial_Nz)

    if optimal_Nz != initial_Nz
        if verbose
            @info "Adjusting Nz from $initial_Nz to $optimal_Nz for optimal factorization"
        end

        # Objective function: minimize squared difference between actual and target Nz
        objective(stretching_factor) = (count_grid_points(stretching_factor, dz, H, Lz) - optimal_Nz)^2

        # Find optimal stretching factor using 1D bounded optimization
        result = optimize(objective, search_bounds[1], search_bounds[2], GoldenSection())
        optimal_stretching_factor = result.minimizer

        if verbose
            @info "Found optimal stretching factor: $optimal_stretching_factor"
        end

        # Create new z-coordinates with optimal stretching factor
        z_coords = create_stretched_z_coordinates(dz, H, Lz, optimal_stretching_factor, min_stretching_factor)

        final_Nz = length(z_coords) - 1
        if verbose
            @info "Adjusted z-grid now has $(length(z_coords)) points (Nz = $final_Nz, target was $optimal_Nz)"
        end
    else
        if verbose
            @info "Initial Nz = $initial_Nz is already optimal for given prime factors"
        end
    end

    return z_coords
end
#---

#+++ Useful GPU show methods
"""
    @CUDAstats ex

A macro that wraps `CUDA.@time` but only executes the timing functionality when CUDA is functional.
If CUDA is not available or not functional, it simply executes the expression without timing.

This is useful for code that should work both with and without GPU support.
"""
macro CUDAstats(ex)
    quote
        if functional()
            @time $(esc(ex))
        else
            $(esc(ex))
        end
    end
end

function get_gpu_memory_usage(gpu_device)
    total_mem = totalmem(gpu_device) |> Float64
    free_mem  = available_memory()
    used_mem  = total_mem - free_mem
    return total_mem, free_mem, used_mem
end

function show_gpu_status()
    # Check if CUDA is available
    if !functional()
        return
    end

    # Get number of available GPUs
    num_devices = length(devices())

    println("="^70)
    println("GPU Status Report")
    println("="^70)
    println("Number of GPUs available: $num_devices")
    println()

    # Iterate through all available GPUs
    for (i, gpu_device) in enumerate(devices())
        # Set current device
        device!(gpu_device)

        # Get device information
        gpu_name  = name(gpu_device)
        total_mem, free_mem, used_mem = get_gpu_memory_usage(gpu_device)

        # Convert to GB for readability
        used_gb = used_mem / (1024^3)
        total_gb = total_mem / (1024^3)
        usage_percent = (used_mem / total_mem) * 100

        # Display information
        println("GPU $i: $gpu_name")
        println("  Used Memory:  $(round(used_gb, digits=2)) GB")
        println("  Total Memory: $(round(total_gb, digits=2)) GB")
        println("  Usage:        $(round(usage_percent, digits=1))%")

        # Add a visual progress bar
        bar_length = 30
        filled_length = Int(round(usage_percent / 100 * bar_length))
        bar = "█" ^ filled_length * "░" ^ (bar_length - filled_length)
        println("  [$(bar)] $(round(usage_percent, digits=1))%")
        println()
        println("Double check with CUDA's native function:")
        memory_status()
    end

    println("=" ^ 70)
end
#---

#+++ Auxiliary immersed boundary metrics
using Oceananigans: Center, Face
using Oceananigans.Fields: @compute
import Oceananigans.Grids: xnode, ynode, znode
using Adapt

const c = Center()
const f = Face()

xnode(i, grid, ℓx) = xnode(i, 1, 1, grid, ℓx, c, c)
ynode(j, grid, ℓy) = ynode(1, j, 1, grid, c, ℓy, c)
znode(k, grid, ℓz) = znode(1, 1, k, grid, c, c, ℓz)

@inline z_distance_from_seamount_boundary_ccc(i, j, k, grid, args...) = znode(k, grid, c) - grid.immersed_boundary.bottom_height[i, j, 1]

struct DistanceCondition{FT}
    from_bottom :: FT
    from_top    :: FT
    from_east   :: FT
end

function DistanceCondition(FT=Float64; from_bottom=5meters, from_top=0, from_east=0)
    from_bottom = convert(FT, from_bottom)
    from_top    = convert(FT, from_top)
    from_east   = convert(FT, from_east)
    return DistanceCondition(from_bottom, from_top, from_east)
end

# Necessary for GPU
Adapt.adapt_structure(to, dc::DistanceCondition) = DistanceCondition(adapt(to, dc.from_bottom),
                                                                     adapt(to, dc.from_top),
                                                                     adapt(to, dc.from_east))

z_distance_from_bottom(args...) = z_distance_from_seamount_boundary_ccc(args...)
z_distance_from_top(i, j, k, grid, args...) = znode(grid.Nz + 1, grid, f) - znode(k, grid, c)
x_distance_from_east(i, j, k, grid, args...) = xnode(grid.Nx + 1, grid, f) - xnode(i, grid, c)
(dc::DistanceCondition)(i, j, k, grid, co) = (z_distance_from_bottom(i, j, k, grid) > dc.from_bottom) &
                                             (z_distance_from_top(i, j, k, grid) > dc.from_top) &
                                             (x_distance_from_east(i, j, k, grid) > dc.from_east)

function measure_FWHM(x, y, elevation)
    H = maximum(elevation)
    Δx = diff(x)[1]
    Δy = diff(y)[1]
    area_at_HM = (elevation .> H/2) .* Δx .* Δy
    FWHM = 2 * √(sum(area_at_HM) / π)
    return FWHM
end
#---
