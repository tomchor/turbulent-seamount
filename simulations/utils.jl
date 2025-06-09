using Parameters
using ImageFiltering: imfilter, Kernel

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
