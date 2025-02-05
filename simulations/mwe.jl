using Oceananigans
using Oceananigans.Units
using PrettyPrinting
using TickTock

grid = RectilinearGrid(size = (4, 4, 4), extent=(1, 1, 1))
model = NonhydrostaticModel(; grid)

simulation = Simulation(model, Δt = 1, stop_time = 10,)
simname = "tokara_res=8_α=0.05_Ro_h=1.4_Fr_h=0.6"
simulation.output_writers[:chk_writer] = Checkpointer(model;
                                                      prefix = simname,
                                                      schedule = TimeInterval(2),
                                                      overwrite_existing = true,
                                                      cleanup = true,
                                                      verbose = true,
                                                      )
run!(simulation, pickup=true)
