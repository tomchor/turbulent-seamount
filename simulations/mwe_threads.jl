using Oceananigans
using Printf
using JLD2
using Oceananigans.Models.NonhydrostaticModels: ConjugateGradientPoissonSolver, FFTBasedPoissonSolver, FourierTridiagonalPoissonSolver
using Oceananigans.Models.NonhydrostaticModels: nonhydrostatic_pressure_solver
using Oceananigans.Solvers: DiagonallyDominantPreconditioner, compute_laplacian!
using Oceananigans.Grids: with_number_type, XYZRegularRG
using Oceananigans.Utils: launch!
using KernelAbstractions: @kernel, @index
using Oceananigans.Architectures: architecture
using Oceananigans.Operators
using Statistics
using Random
using CUDA

function initial_conditions!(model)
    h = 0.05
    x₀ = 0.5
    y₀ = 0.5
    z₀ = 0.55
    bᵢ(x, y, z) = - exp(-((x - x₀)^2 + (y - y₀)^2 + (z - z₀)^2) / 2h^2)

    set!(model, b=bᵢ)
end

function setup_grid(Nx, Ny, Nz, arch)
    grid = RectilinearGrid(arch, Float64,
                        size = (Nx, Ny, Nz), 
                        halo = (4, 4, 4),
                        x = (0, 1),
                        y = (0, 1),
                        z = (0, 1),
                        topology = (Bounded, Bounded, Bounded))

    slope(x, y) = (5 + tanh(40*(x - 1/6)) + tanh(40*(x - 2/6)) + tanh(40*(x - 3/6)) + tanh(40*(x - 4/6)) + tanh(40*(x - 5/6))) / 20 + 
                  (5 + tanh(40*(y - 1/6)) + tanh(40*(y - 2/6)) + tanh(40*(y - 3/6)) + tanh(40*(y - 4/6)) + tanh(40*(y - 5/6))) / 20

    # grid = ImmersedBoundaryGrid(grid, GridFittedBottom(slope))
    grid = ImmersedBoundaryGrid(grid, PartialCellBottom(slope))
    return grid
end

function setup_model(grid, pressure_solver)
    model = NonhydrostaticModel(; grid, pressure_solver,
                                  advection = WENO(),
                                  coriolis = FPlane(f=0.1),
                                  tracers = :b,
                                  buoyancy = BuoyancyTracer())

    initial_conditions!(model)
    return model
end

function setup_simulation(model)
    Δt = 1e-3
    simulation = Simulation(model; Δt = Δt, stop_iteration = 10)
    
    wall_time = Ref(time_ns())

    d = Field{Center, Center, Center}(grid)

    function progress(sim)
        pressure_solver = sim.model.pressure_solver
    
        if pressure_solver isa ConjugateGradientPoissonSolver
            pressure_iters = iteration(pressure_solver)
        else
            pressure_iters = 0
        end

        msg = @sprintf("iter: %d, time: %s, Δt: %.4f, Poisson iters: %d",
                        iteration(sim), prettytime(time(sim)), sim.Δt, pressure_iters)
        elapsed = 1e-9 * (time_ns() - wall_time[])
        msg *= @sprintf(", max u: %6.3e, max v: %6.3e, max w: %6.3e, max b: %6.3e, max d: %6.3e, max pressure: %6.3e, wall time: %s",
                        maximum(sim.model.velocities.u),
                        maximum(sim.model.velocities.v),
                        maximum(sim.model.velocities.w),
                        maximum(sim.model.tracers.b),
                        maximum(d),
                        maximum(sim.model.pressures.pNHS),
                        prettytime(elapsed))
    
        @info msg
        wall_time[] = time_ns()
    
        return nothing
    end

    simulation.callbacks[:progress] = Callback(progress, IterationInterval(1))

    return simulation
end

arch = GPU()
Nx = Ny = Nz = 32
grid = setup_grid(Nx, Ny, Nz, arch)

pressure_solver = ConjugateGradientPoissonSolver(grid, maxiter=10000; preconditioner=nothing)
model = setup_model(grid, pressure_solver)

simulation = setup_simulation(model)

run!(simulation)
