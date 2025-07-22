if ("PBS_JOBID" in keys(ENV))  @info "Job ID" ENV["PBS_JOBID"] end # Print job ID if this is a PBS simulation
#using Pkg; Pkg.instantiate()
using InteractiveUtils
versioninfo()
using ArgParse
using CUDA: has_cuda_gpu
using PrettyPrinting: pprintln
using TickTock: tick, tock
using NCDatasets: NCDataset
using Interpolations: LinearInterpolation

using Oceananigans
using Oceananigans.Units
using Oceananigans: on_architecture
using Oceananigans.TurbulenceClosures: Smagorinsky, DynamicCoefficient, LagrangianAveraging, DynamicSmagorinsky
using Oceananigans.OutputWriters: write_output!

#+++ Parse inital arguments
"Returns a dictionary of command line arguments."
function parse_command_line_arguments()
    settings = ArgParseSettings()
    @add_arg_table! settings begin

        "--simname"
            help = "Simulation name for output"
            default = "seamount"
            arg_type = String

        "--x₀"
            default = 0
            arg_type = Float64

        "--y₀"
            default = 0
            arg_type = Float64

        "--aspect"
            help = "Desired cell aspect ratio; Δx/Δz = Δy/Δz"
            default = 2
            arg_type = Float64

        "--dz"
            default = 8
            arg_type = Int

        "--V∞"
            default = 0.1meters/second
            arg_type = Float64

        "--H"
            default = 110meters
            arg_type = Float64

        "--L"
            help = "Scale for smoothing the bathymetry"
            default = 0meters
            arg_type = Float64

        "--Ro_h"
            default = 1.25
            arg_type = Float64

        "--Fr_h"
            default = 0.2
            arg_type = Float64

        "--Lx_ratio"
            default = 6 # Lx / FWHM
            arg_type = Float64

        "--Ly_ratio"
            default = 10 # Ly / FWHM
            arg_type = Float64

        "--Lz_ratio"
            default = 2 # Lz / H
            arg_type = Float64

        "--Rz"
            default = 2.5e-4
            arg_type = Float64

        "--closure"
            default = "AMD"
            arg_type = String

        "--runway_length_fraction_FWHM"
            default = 3.5 # y_offset / FWHM (how far from the inflow the headland is)
            arg_type = Float64

        "--T_advective_spinup"
            default = 10 # Should be a multiple of interval_time_avg
            arg_type = Float64

        "--T_advective_statistics"
            default = 10 # Should be a multiple of interval_time_avg
            arg_type = Float64

    end
    return parse_args(settings, as_symbols=true)
end

params = (; parse_command_line_arguments()...)
rundir = @__DIR__
#---

#+++ Figure out architecture
if has_cuda_gpu()
    arch = GPU()
else
    arch = CPU()
    params = (; params..., dz = 50meters)
end
@info "Starting simulation $(params.simname) with a dividing factor of $(params.dz) and a $arch architecture\n"
#---

#+++ Get bathymetry file, z_coords, and secondary simulation parameters
ds_bathymetry = NCDataset(joinpath(@__DIR__, "../bathymetry/balanus-bathymetry-preprocessed.nc"))

let
    #+++ Geometry
    H_ratio = params.H / ds_bathymetry.attrib["H"]
    FWHM = ds_bathymetry.attrib["FWHM"] * H_ratio
    α = params.H / FWHM

    Lx = params.Lx_ratio * FWHM
    Ly = params.Ly_ratio * FWHM
    Lz = params.Lz_ratio * params.H

    y_offset = params.runway_length_fraction_FWHM * FWHM
    #---

    global params = merge(params, Base.@locals)
end

include("$(@__DIR__)/utils.jl")
z_coords = create_optimal_z_coordinates(params.dz, params.H, params.Lz, (2, 3, 5),
                                        initial_stretching_factor = 1.05)

let
    #+++ Simulation size
    Nx = max(ceil(Int, params.Lx / (params.aspect * params.dz)), 5)
    Ny = max(ceil(Int, params.Ly / (params.aspect * params.dz)), 5)

    Nx = closest_factor_number((2, 3, 5), Nx)
    Ny = closest_factor_number((2, 3, 5), Ny)
    Nz = length(z_coords) - 1
    N_total = Nx * Ny * Nz
    #---

    #+++ Dynamically-relevant secondary parameters
    f₀ = f_0 = params.V∞ / (params.Ro_h * params.FWHM)
    N²∞ = N2_inf = (params.V∞ / (params.Fr_h * params.H))^2
    R1 = √N²∞ * params.H / f₀
    z₀ = z_0 = params.Rz * params.H
    #---

    #+++ Diagnostic parameters
    Γ = params.α * params.Fr_h # nonhydrostatic parameter (Schar 2002)
    Bu_h = (params.Ro_h / params.Fr_h)^2
    Slope_Bu = params.Ro_h / params.Fr_h # approximate slope Burger number
    @assert Slope_Bu ≈ params.α * √N²∞ / f₀
    #---

    #+++ Time scales
    T_inertial = 2π / f₀
    T_cycle = params.Ly / params.V∞
    T_advective = params.FWHM / params.V∞
    #---

    global params = merge(params, Base.@locals)
end
pprintln(params)
#---

#+++ Base grid
grid_base = RectilinearGrid(arch; topology = (Periodic, Bounded, Bounded),
                            size = (params.Nx, params.Ny, params.Nz),
                            x = (-params.Lx/2, +params.Lx/2),
                            y = (-params.y_offset, params.Ly - params.y_offset),
                            z = z_coords,
                            halo = (4, 4, 4),
                            )
@info grid_base
params = (; params..., Δz_min = minimum_zspacing(grid_base))
#---

#+++ Interpolate (and maybe smooth) bathymetry
shrunk_elevation = ds_bathymetry["periodic_elevation"] * params.H_ratio
shrunk_x = ds_bathymetry["x"] * params.H_ratio
shrunk_y = ds_bathymetry["y"] * params.H_ratio

@info "Interpolating bathymetry"
itp = LinearInterpolation((shrunk_x, shrunk_y), shrunk_elevation,  extrapolation_bc=0)

x_grid = xnodes(grid_base, Center(), Center(), Center())
y_grid = ynodes(grid_base, Center(), Center(), Center())
interpolated_bathymetry_cpu = itp.(x_grid, reshape(y_grid, (1, grid_base.Ny)))

if params.L == 0
    @warn "No smoothing performed on the bathymetry"
    final_bathymetry_cpu = interpolated_bathymetry_cpu
else
    @warn "Smoothing bathymetry with length scale $(params.L)"
    final_bathymetry_cpu = smooth_bathymetry(interpolated_bathymetry_cpu, grid_base, scale_x=params.L, scale_y=params.L, bc_x="circular", bc_y="replicate")
end

final_bathymetry = on_architecture(grid_base.architecture, final_bathymetry_cpu)
#---

#+++ Immersed boundary
PCB = GridFittedBottom(final_bathymetry)

grid = ImmersedBoundaryGrid(grid_base, PCB)
@info grid
#---

#+++ Drag (Implemented as in https://doi.org/10.1029/2005WR004685)
z₀ = params.z_0 # roughness length
z₁ = minimum_zspacing(grid_base, Center(), Center(), Center())/2
@info "Using z₁ =" z₁

const κᵛᵏ = 0.4 # von Karman constant
params = (; params..., c_dz = (κᵛᵏ / log(z₁/z₀))^2) # quadratic drag coefficient
@info "Defining momentum BCs with Cᴰ (x, y, z) =" params.c_dz

@inline τᵘ_drag(x, y, z, t, u, v, w, p) = -p.Cᴰ * u * √(u^2 + v^2 + w^2)
@inline τᵛ_drag(x, y, z, t, u, v, w, p) = -p.Cᴰ * v * √(u^2 + v^2 + w^2)
@inline τʷ_drag(x, y, z, t, u, v, w, p) = -p.Cᴰ * w * √(u^2 + v^2 + w^2)

τᵘ = FluxBoundaryCondition(τᵘ_drag, field_dependencies = (:u, :v, :w), parameters=(; Cᴰ = params.c_dz,))
τᵛ = FluxBoundaryCondition(τᵛ_drag, field_dependencies = (:u, :v, :w), parameters=(; Cᴰ = params.c_dz,))
τʷ = FluxBoundaryCondition(τʷ_drag, field_dependencies = (:u, :v, :w), parameters=(; Cᴰ = params.c_dz,))
#---

#+++ Open boundary conditions for velocitities
using Oceananigans.BoundaryConditions: PerturbationAdvectionOpenBoundaryCondition

u_south = u_north = ValueBoundaryCondition(0)

v_south = OpenBoundaryCondition(params.V∞)
v_north = PerturbationAdvectionOpenBoundaryCondition(params.V∞; inflow_timescale = 2minutes, outflow_timescale = 2minutes,)

w_south = w_north = ValueBoundaryCondition(0)
#---

#+++ Boundary and initial conditions for buoyancy
struct LinearStratification
    N²∞ :: Float64 # stratification strength (s⁻²)
end

(strat::LinearStratification)(z) = strat.N²∞ * z
(strat::LinearStratification)(x, y, z) = strat(z) # For initial condition
(strat::LinearStratification)(x, y, z, t) = strat(z) # For the sponge layer
b∞ = LinearStratification(params.N²∞)

b_boundaries(x, z, t, N²∞) = z * N²∞
b_south = b_north = ValueBoundaryCondition(b_boundaries, parameters=params.N²∞)
#---

#+++ Assemble BCs
u_bcs = FieldBoundaryConditions(south=u_south, north=u_north, immersed=τᵘ)
v_bcs = FieldBoundaryConditions(south=v_south, north=v_north, immersed=τᵛ)
w_bcs = FieldBoundaryConditions(south=w_south, north=w_north, immersed=τʷ)
b_bcs = FieldBoundaryConditions(south=b_south, north=b_north)

bcs = (u=u_bcs, v=v_bcs, w=w_bcs, b=b_bcs)
#---

#+++ Define geostrophic forcing
@inline geostrophy(x, y, z, t, p) = -p.f₀ * p.V∞
Fᵤ = Forcing(geostrophy, parameters = (; params.f₀, params.V∞))
#---

#+++ Turbulence closure
if params.closure == "CSM"
    closure = SmagorinskyLilly(C=0.13, Pr=1)
elseif params.closure == "DSM"
    closure = Smagorinsky(coefficient=DynamicCoefficient(averaging=LagrangianAveraging(), schedule=IterationInterval(5)), Pr=1)
elseif params.closure == "AMD"
    closure = AnisotropicMinimumDissipation(C=1/12)
elseif params.closure == "AMC"
    include("AMD.jl")
    closure = AnisotropicMinimumDissipation(C=1/12)
elseif params.closure == "NON"
    closure = nothing
else
    throw(ArgumentError("Check options for `closure`"))
end

#---

#+++ Add top sponge layer
let
    h_sponge = 0.2 * params.Lz
    sponge_damping_rate = max(√params.N²∞, params.α * params.V∞ / h_sponge) / 10

    global params = merge(params, Base.@locals)
end

mask_top = PiecewiseLinearMask{:z}(center=params.Lz, width=params.h_sponge)
w_sponge = Relaxation(rate=params.sponge_damping_rate, mask=mask_top, target=0)
v_sponge = Relaxation(rate=params.sponge_damping_rate, mask=mask_top, target=params.V∞)
b_sponge = Relaxation(rate=params.sponge_damping_rate, mask=mask_top, target=b∞)
#---

#+++ Model and ICs
@info "Creating model"
model = NonhydrostaticModel(grid = grid, timestepper = :RungeKutta3,
                            advection = WENO(order=5),
                            buoyancy = BuoyancyTracer(),
                            coriolis = FPlane(params.f_0),
                            tracers = :b,
                            closure = closure,
                            boundary_conditions = bcs,
                            forcing = (; u=Fᵤ, v=v_sponge, w=w_sponge, b=b_sponge),
                            hydrostatic_pressure_anomaly = CenterField(grid),
                            )
@info "" model
show_gpu_status()

set!(model, b=(x, y, z) -> b∞(z), v=params.V∞)
#---

#+++ Create simulation
params = (; params..., T_advective_max = params.T_advective_spinup + params.T_advective_statistics)
simulation = Simulation(model, Δt = 0.2 * minimum_zspacing(grid.underlying_grid) / params.V∞,
                        stop_time = params.T_advective_max * params.T_advective,
                        wall_time_limit = 23hours,
                        minimum_relative_step = 1e-10,
                        )

using Oceanostics.ProgressMessengers
walltime_per_timestep = StepDuration(with_prefix=false) # This needs to instantiated here, and not in the function below
walltime = Walltime()
progress(simulation) = @info (PercentageProgress(with_prefix=false, with_units=false)
                              + "$(round(time(simulation)/params.T_advective; digits=2)) adv periods" + walltime
                              + TimeStep() + "CFL = " * AdvectiveCFLNumber(with_prefix=false)
                              + "step dur = " * walltime_per_timestep)(simulation)
simulation.callbacks[:progress] = Callback(progress, IterationInterval(40))

conjure_time_step_wizard!(simulation, IterationInterval(1), max_change=1.05, cfl=0.9, min_Δt=1e-4, max_Δt=1/√params.N²∞)

t_switch = 12 * params.T_advective
function cfl_changer(sim)
    if sim.model.clock.time > 0
        @warn "Changing target cfl"
        simulation.callbacks[:time_step_wizard].func.cfl = 0.8
    end
end
add_callback!(simulation, cfl_changer, SpecifiedTimes([t_switch]); name=:cfl_changer)

@info "" simulation
#---

#+++ Outputs and diagnostics
include("$rundir/diagnostics.jl")

#+++ Define checkpointer/pickup
write_ckpt = params.dz < 2
interval_time_avg = params.T_advective

if write_ckpt
    checkpointer_prefix = "ckpt.$(params.simname)"
    if any(startswith(checkpointer_prefix), readdir("data"))
        @warn "Checkpoint for $(params.simname) found. Assuming this is a pick-up simulation! Setting overwrite_existing=false."
        overwrite_existing = false
    else
        @warn "No checkpoint for $(params.simname) found. Setting overwrite_existing=true."
        overwrite_existing = true
    end

    #+++ Construct checkpointer
    @info "Setting up checkpointer"
    simulation.output_writers[:ckpt_writer] = checkpointer = @CUDAstats Checkpointer(model;
                                                                                     dir = "$rundir/data/",
                                                                                     prefix = checkpointer_prefix,
                                                                                     schedule = TimeInterval(interval_time_avg),
                                                                                     overwrite_existing = true,
                                                                                     cleanup = true,
                                                                                     )
    #---

else
    @warn "No checkpointing necessary for this simulation."
    overwrite_existing = true
end
#---

tick()
construct_outputs(simulation;
                  simname = params.simname,
                  rundir = rundir,
                  params = params,
                  overwrite_existing = overwrite_existing,
                  interval_2d = 0.1*params.T_advective,
                  interval_3d = 0.5*params.T_advective,
                  interval_time_avg,
                  write_xyzi = true,
                  write_xizi = false,
                  write_xyii = true,
                  write_iyzi = true,
                  write_xyza = false,
                  write_xyia = false,
                  write_aaai = true,
                  write_ckpt,
                  debug = false,
                  )
tock()
#---

#+++ Run simulations and plot video afterwards
show_gpu_status()
@info "Starting simulation"
run!(simulation, pickup=write_ckpt)
#---

#+++ Write final checkpoint to disk if checkpointer exists
# if write_ckpt
#     @info "Writing final checkpoint to disk"
#     write_output!(checkpointer, model)
# end
#---

#+++ Plot video
include("$rundir/plot_2d_animation.jl")
include("$rundir/plot_3d_animation.jl")
#---
