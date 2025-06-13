using Parameters
using ImageFiltering: imfilter, Kernel
using Optim: GoldenSection, optimize

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

#+++ Smooth bathymetry
function smooth_bathymetry(elevation; window_size_x, window_size_y, bc_x="circular", bc_y="replicate")
    # Create separate Gaussian kernels for x and y directions
    kernel_x = Kernel.gaussian((window_size_x, 0))
    kernel_y = Kernel.gaussian((0, window_size_y))

    # Apply filters sequentially with different boundary conditions
    # First smooth in x direction
    smoothed_x = imfilter(elevation, kernel_x, bc_x)
    # Then smooth in y direction
    smoothed = imfilter(smoothed_x, kernel_y, bc_y)

    return smoothed
end

function smooth_bathymetry(elevation, grid; scale_x, scale_y, bc_x="circular", bc_y="replicate")
    # Get minimum grid spacing in x and y directions
    Δx_min = minimum_xspacing(grid)
    Δy_min = minimum_yspacing(grid)

    # Convert physical scales to window sizes
    # We use ceil to ensure we have enough points to cover the scale
    # The factor of 2 is because the Gaussian kernel's standard deviation
    # should be about half the desired smoothing length
    window_size_x = scale_x / (2 * Δx_min)
    window_size_y = scale_y / (2 * Δy_min)

    # Call the original method with calculated window sizes
    return smooth_bathymetry(elevation; window_size_x, window_size_y, bc_x, bc_y)
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
using CUDA: devices, device!, functional, totalmem, name, available_memory, memory_status
function show_gpu_status()
    # Check if CUDA is available
    if !functional()
        println("CUDA is not available on this system")
        return
    end

    # Get number of available GPUs
    num_devices = length(devices())

    println("=" ^ 70)
    println("GPU Status Report")
    println("=" ^ 70)
    println("Number of GPUs available: $num_devices")
    println()

    # Iterate through all available GPUs
    for (i, device) in enumerate(devices())
        # Set current device
        device!(device)

        # Get device information
        gpu_name  = name(device)
        total_mem = totalmem(device)
        free_mem  = available_memory()
        used_mem  = total_mem - free_mem

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

#+++ Piecewise Linear Mask
"""
    PiecewiseLinearMask{D}(center, width)

Callable object that returns a piecewise linear masking function centered on
`center`, with `width`, and varying along direction `D`. The mask is:
- 0 when |D - center| > width
- 1 when D = center
- Linear interpolation between 0 and 1 when |D - center| ≤ width

Example
=======

Create a piecewise linear mask centered on `z=0` with width `1` meter.

```julia
julia> mask = PiecewiseLinearMask{:z}(center=0, width=1)
```
"""
struct PiecewiseLinearMask{D, T}
    center :: T
     width :: T

    function PiecewiseLinearMask{D}(; center, width) where D
        T = promote_type(typeof(center), typeof(width))
        return new{D, T}(center, width)
    end
end

@inline function (p::PiecewiseLinearMask{:x})(x, y, z)
    d = abs(x - p.center)
    return d > p.width ? 0.0 : 1.0 - d/p.width
end

@inline function (p::PiecewiseLinearMask{:y})(x, y, z)
    d = abs(y - p.center)
    return d > p.width ? 0.0 : 1.0 - d/p.width
end

@inline function (p::PiecewiseLinearMask{:z})(x, y, z)
    d = abs(z - p.center)
    return d > p.width ? 0.0 : 1.0 - d/p.width
end

Base.summary(p::PiecewiseLinearMask{D}) where D =
    "piecewise_linear($D, center=$(p.center), width=$(p.width))"
#---
