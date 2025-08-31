if ("PBS_JOBID" in keys(ENV))  @info "Job ID" ENV["PBS_JOBID"] end # Print job ID if this is a PBS simulation
#using Pkg; Pkg.instantiate()
using InteractiveUtils
versioninfo()
using ArgParse
using CUDA: has_cuda_gpu
using PrettyPrinting: pprintln
using TickTock: tick, tock
using NCDatasets: NCDataset
import Interpolations

using Oceananigans
using Oceananigans.Units
using Oceananigans: on_architecture
using Oceananigans.TurbulenceClosures: Smagorinsky, DynamicCoefficient, LagrangianAveraging, DynamicSmagorinsky
using Oceananigans.Solvers: ConjugateGradientPoissonSolver, fft_poisson_solver

include("$(@__DIR__)/utils.jl")

#+++ Parse inital arguments
"Returns a dictionary of command line arguments."
function parse_command_line_arguments()
    settings = ArgParseSettings()
    @add_arg_table! settings begin

        "--simname"
            help = "Simulation name for output"
            default = "seamount"
            arg_type = String

        "--dz"
            default = 8meters
            arg_type = Int

        "--U∞"
            default = 0.1meters/second
            arg_type = Float64

        "--H"
            default = 100meters
            arg_type = Float64

        "--FWHM"
            help = "Full width at half maximum of the seamount"
            default = 500meters
            arg_type = Float64

        "--L"
            help = "Scale for smoothing the bathymetry (as a ratio of FWHM)"
            default = 0.2
            arg_type = Float64

        "--Ro_h"
            default = 0.2
            arg_type = Float64

        "--Fr_h"
            default = 1.25
            arg_type = Float64

        "--Lx"
            help = "Domain length in x-direction"
            default = 3500meters
            arg_type = Float64

        "--Ly"
            help = "Domain length in y-direction"
            default = 2000meters
            arg_type = Float64

        "--Lz_ratio"
            default = 2 # Lz / H
            arg_type = Float64

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

        "--Rz"
            default = 2.5e-4
            arg_type = Float64

        "--closure"
            default = "DSM"
            arg_type = String

        "--runway_length_fraction_FWHM"
            default = 2 # x_offset / FWHM (how far from the inflow the headland is)
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

#+++ Figure out architecture (and maybe change dz)
if has_cuda_gpu()
    arch = GPU()
else
    arch = CPU()
    params = (; params..., dz = 50meters)
end
@info "Starting simulation $(params.simname) with a vertical spacing of $(params.dz) meters and $arch architecture\n"
#---

#+++ Create interpolant for (and maybe smooth) bathymetry
ds_bathymetry = NCDataset(joinpath(@__DIR__, "../bathymetry/balanus-bathymetry-preprocessed.nc"))
elevation = ds_bathymetry["periodic_elevation"]
x = ds_bathymetry["x"]
y = ds_bathymetry["y"]

original_FWHM = measure_FWHM(x, y, elevation)
@assert isapprox(original_FWHM, ds_bathymetry.attrib["FWHM"], rtol=1e-3)

params = (; params..., H_ratio = params.H / maximum(elevation), # How much do we rescale in the vertical?
                       FWHM_ratio = params.FWHM / ds_bathymetry.attrib["FWHM"]) # How much do we rescale in the horizontal?
shrunk_elevation = ds_bathymetry["periodic_elevation"] .* params.H_ratio # Rescale the elevation to the new height

if params.L == 0
    @warn "No smoothing performed on the bathymetry"
    shrunk_smoothed_elevation = shrunk_elevation
else
    @warn "Smoothing bathymetry with length scale L/FWHM=$(params.L)"
    shrunk_smoothed_elevation = smooth_bathymetry(shrunk_elevation, x, y;
                                                  scale_x = params.L * ds_bathymetry.attrib["FWHM"], # Based on the data's FWHM
                                                  scale_y = params.L * ds_bathymetry.attrib["FWHM"], # Based on the data's FWHM
                                                  bc_x="circular",
                                                  bc_y="replicate",)
end

params = (; params..., H_after_smoothing = maximum(shrunk_smoothed_elevation))

# Rescale the horizontal dimensions to the new FWHM.
# Note that the smoothed bathymetry is likely a bit shorter than the original, and we do not correct for that on purpose.
shrunk_x = x .* params.FWHM_ratio
shrunk_y = y .* params.FWHM_ratio

@info "Interpolating bathymetry"
bathymetry_itp = Interpolations.LinearInterpolation((shrunk_x, shrunk_y), shrunk_smoothed_elevation, extrapolation_bc=Interpolations.Flat())
close(ds_bathymetry)
#---

#+++ Get domain sizes, z_coords, and secondary simulation parameters
let
    #+++ Geometry
    α = params.H / params.FWHM
    Lz = params.Lz_ratio * params.H

    x_offset = params.runway_length_fraction_FWHM * params.FWHM
    L_meters = params.L * params.FWHM  # Convert dimensionless L to meters
    #---

    global params = merge(params, Base.@locals)
end

z_coords = create_optimal_z_coordinates(params.dz, params.H, params.Lz, (2, 3, 5), initial_stretching_factor = 1.05)

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
    f₀ = f_0 = params.U∞ / (params.Ro_h * params.FWHM)
    N²∞ = N2_inf = (params.U∞ / (params.Fr_h * params.H))^2
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
    T_cycle = params.Lx / params.U∞
    T_advective = params.FWHM / params.U∞
    #---

    global params = merge(params, Base.@locals)
end

pprintln(params)
#---

#+++ Base grid
grid_base = RectilinearGrid(arch; topology = (Bounded, Periodic, Bounded),
                            size = (params.Nx, params.Ny, params.Nz),
                            x = (-params.x_offset, params.Lx - params.x_offset),
                            y = (-params.Ly/2, +params.Ly/2),
                            z = z_coords,
                            halo = (4, 4, 4),
                            )
@info grid_base
params = (; params..., Δz_min = minimum_zspacing(grid_base))
#---

#+++ Interpolate bathymetry and create immersed boundary grid
x_grid = xnodes(grid_base, Center(), Center(), Center())
y_grid = ynodes(grid_base, Center(), Center(), Center())
interpolated_bathymetry_cpu = bathymetry_itp.(reshape(y_grid, (1, grid_base.Ny)), reshape(x_grid, (grid_base.Nx, 1)))
interpolated_bathymetry = on_architecture(grid_base.architecture, interpolated_bathymetry_cpu)

grid = ImmersedBoundaryGrid(grid_base, GridFittedBottom(interpolated_bathymetry))
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

u_west = OpenBoundaryCondition(params.U∞)
u_east = PerturbationAdvectionOpenBoundaryCondition(params.U∞; inflow_timescale = 2minutes, outflow_timescale = 30minutes)

v_west = w_west = ValueBoundaryCondition(0)
v_east = w_east = FluxBoundaryCondition(0)
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
b_west = b_east = ValueBoundaryCondition(b_boundaries, parameters=params.N²∞)
#---

#+++ Assemble BCs
u_bcs = FieldBoundaryConditions(west=u_west, east=u_east, immersed=τᵘ)
v_bcs = FieldBoundaryConditions(west=v_west, east=v_east, immersed=τᵛ)
w_bcs = FieldBoundaryConditions(west=w_west, east=w_east, immersed=τʷ)
b_bcs = FieldBoundaryConditions(west=b_west, east=b_east)

bcs = (u=u_bcs, v=v_bcs, w=w_bcs, b=b_bcs)
#---

#+++ Define geostrophic forcing
@inline geostrophy(x, y, z, t, p) = p.f₀ * p.U∞
Fᵥ = Forcing(geostrophy, parameters = (; params.f₀, params.U∞))
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
    sponge_damping_rate = max(√params.N²∞, params.α * params.U∞ / h_sponge) / 10

    global params = merge(params, Base.@locals)
end

mask_top = PiecewiseLinearMask{:z}(center=params.Lz, width=params.h_sponge)
w_sponge = Relaxation(rate=params.sponge_damping_rate, mask=mask_top, target=0)
u_sponge = Relaxation(rate=params.sponge_damping_rate, mask=mask_top, target=params.U∞)
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
                            forcing = (; u=u_sponge, v=Fᵥ, w=w_sponge, b=b_sponge),
                            hydrostatic_pressure_anomaly = CenterField(grid),
                            #pressure_solver = ConjugateGradientPoissonSolver(grid, preconditioner = fft_poisson_solver(grid.underlying_grid), maxiter = 100),
                            )
@info "" model
show_gpu_status()

set!(model, b=(x, y, z) -> b∞(z), u=params.U∞)
#---

#+++ Create simulation
params = (; params..., T_advective_max = params.T_advective_spinup + params.T_advective_statistics)
simulation = Simulation(model, Δt = 0.2 * minimum_zspacing(grid.underlying_grid) / params.U∞,
                        stop_time = params.T_advective_max * params.T_advective,
                        wall_time_limit = 23hours,
                        minimum_relative_step = 1e-10,
                        )

using Oceanostics.ProgressMessengers
walltime_per_timestep = StepDuration(with_prefix=false) # This needs to instantiated here, and not in the function below
walltime = Walltime()
cg_iterations(simulation) = simulation.model.pressure_solver isa ConjugateGradientPoissonSolver ? "iterations = $(iteration(model.pressure_solver))" : ""
progress(simulation) = @info (PercentageProgress(with_prefix=false, with_units=false)
                              + "$(round(time(simulation)/params.T_advective; digits=2)) adv periods" + walltime
                              + TimeStep() + "CFL = " * AdvectiveCFLNumber(with_prefix=false)
                              + MaxUVelocity()
                              + "step dur = " * walltime_per_timestep
                              + cg_iterations(simulation)
                              )(simulation)
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
                  write_xizi = true,
                  write_xyii = true,
                  write_iyzi = false,
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

#+++ Plot video
include("$rundir/plot_2d_animation.jl")
include("$rundir/plot_3d_animation.jl")
#---
