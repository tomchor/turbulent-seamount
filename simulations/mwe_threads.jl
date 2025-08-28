using Oceananigans
using Printf
using JLD2
using Oceananigans.Models.NonhydrostaticModels: ConjugateGradientPoissonSolver, FFTBasedPoissonSolver, FourierTridiagonalPoissonSolver
using Oceananigans.Models.NonhydrostaticModels: nonhydrostatic_pressure_solver
using Oceananigans.Operators
using CUDA

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
        msg *= @sprintf(", max u: %6.3e, max v: %6.3e, max w: %6.3e",
                        maximum(sim.model.velocities.u),
                        maximum(sim.model.velocities.v),
                        maximum(sim.model.velocities.w))
    
        @info msg
        wall_time[] = time_ns()
    
        return nothing
    end

    simulation.callbacks[:progress] = Callback(progress, IterationInterval(1))

    return simulation
end

arch = GPU()
N = 32

grid = RectilinearGrid(arch, size = (N, N, N), extent = (1, 1, 1),
topology = (Bounded, Periodic, Bounded))
grid = ImmersedBoundaryGrid(grid, PartialCellBottom(.5))


model = NonhydrostaticModel(; grid,
                                  advection = WENO(),
                                  coriolis = FPlane(f=0.1),
                                  tracers = :b,)


simulation = setup_simulation(model)

run!(simulation)
