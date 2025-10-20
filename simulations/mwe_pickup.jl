using Oceananigans
using CUDA

underlying_grid_gpu = RectilinearGrid(GPU(); size = (8, 8, 8), extent = (1, 1, 1))
grid_gpu = ImmersedBoundaryGrid(underlying_grid_gpu, GridFittedBottom(-1/2))

model_gpu = NonhydrostaticModel(grid = grid_gpu)
simulation_gpu = Simulation(model_gpu, Δt = 0.01, stop_iteration = 10)
checkpointer_gpu = Checkpointer(model_gpu; dir = "data/", prefix = "mwe_pickup", schedule = IterationInterval(5))

simulation_gpu.output_writers[:checkpointer_gpu] = checkpointer_gpu

run!(simulation_gpu, pickup = true)
set!(model_gpu, "data/mwe_pickup_iteration10.jld2")
