using Parameters
using ImageFiltering: imfilter, Kernel

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
