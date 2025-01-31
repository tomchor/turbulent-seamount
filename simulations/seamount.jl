if ("PBS_JOBID" in keys(ENV))  @info "Job ID" ENV["PBS_JOBID"] end # Print job ID if this is a PBS simulation
#using Pkg; Pkg.instantiate()
using InteractiveUtils
versioninfo()
using ArgParse
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: Smagorinsky, DynamicCoefficient, LagrangianAveraging
using PrettyPrinting
using TickTock

using CUDA: @allowscalar, has_cuda_gpu

#+++ Preamble
#+++ Parse inital arguments
"Returns a dictionary of command line arguments."
function parse_command_line_arguments()
    settings = ArgParseSettings()
    @add_arg_table! settings begin

        "--simname"
            help = "Setup and name of simulation in siminfo.jl"
            default = "tokara-f64"
            arg_type = String

        "--x₀"
            default = 0
            arg_type = Number

        "--y₀"
            default = 0
            arg_type = Number

        "--N_max"
            default = 180e6
            arg_type = Number

        "--V∞"
            default = 0.1meters/second

        "--H"
            default = 50meters

        "--α"
            help = "H / FWMH"
            default = 0.05

        "--Ro_h"
            default = 1.4

        "--Fr_h"
            default = 0.6

        "--Lx_ratio"
            default = 4 # Lx / L

        "--Ly_ratio"
            default = 15 # Ly / L

        "--Lz_ratio"
            default = 1.2 # Lz / L

        "--Rz"
            default = 2.5e-4

        "--runway_length_fraction_L"
            default = 4 # y_offset / L (how far from the inflow is the headland)

        "--T_advective_spinup"
            default = 20 # Should be a multiple of 20

        "--T_advective_statistics"
            default = 60 # Should be a multiple of 20
 
    end
    return parse_args(settings, as_symbols=true)
end

params = (; parse_command_line_arguments()...)
rundir = @__DIR__
#---

#+++ Figure out name, dimensions, modifier, etc
sep = "-"
global configname, modifiers... = split(params.simname, sep)
global f2  = "f2"  in modifiers ? true : false
global f4  = "f4"  in modifiers ? true : false
global f8  = "f8"  in modifiers ? true : false
global f16 = "f16" in modifiers ? true : false
global f32 = "f32" in modifiers ? true : false
global f64 = "f64" in modifiers ? true : false
global CSM = "CSM" in modifiers ? true : false # Constan SMagorinsky
global DSM = "DSM" in modifiers ? true : false # Dynamic SMagorinsky
global V2  =  "V2" in modifiers ? true : false
#---

#+++ Modify factor accordingly
if f2
    factor = 2
elseif f4
    factor = 4
elseif f8
    factor = 8
elseif f16
    factor = 16
elseif f32
    factor = 32
elseif f64
    factor = 64
else
    factor = 1
end
params = (; params..., factor)
#---

#+++ Figure out architecture
if has_cuda_gpu()
    arch = GPU()
else
    arch = CPU()
end
@info "Starting simulation $(params.simname) with a dividing factor of $factor and a $arch architecture\n"
#---

#+++ Get primary simulation parameters
include("$(@__DIR__)/siminfo.jl")

let

    #+++ Geometry
    θ_rad = atan(params.α)
    FWMH = params.H / params.α
    L = FWMH / (2√log(2)) # The proper L for an exponential to achieve FWMH

    Lx = params.Lx_ratio * L
    Ly = params.Ly_ratio * L
    Lz = params.Lz_ratio * params.H

    y_offset = params.runway_length_fraction_L * L
    #---

    #+++ Simulation size
    Nx, Ny, Nz = get_sizes(params.N_max ÷ (factor^3); Lx, Ly, Lz, aspect_ratio_x=3.2, aspect_ratio_y=3.2)
    N_total = Nx * Ny * Nz
    #---

    #+++ Dynamically-relevant secondary parameters
    f₀ = f_0 = params.V∞ / (params.Ro_h * FWMH)
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
    T_strouhal = L / (params.V∞ * 0.2)
    T_cycle = Ly / params.V∞
    T_advective = L / params.V∞
    #---

    global params = merge(params, Base.@locals)
end

if V2
    params = (; params..., V∞ = 2*params.V∞)
end
pprintln(params)
#---
#---

#+++ Base grid
refinement = 1.1 # controls spacing near surface (higher means finer spaced)
stretching = 25 # controls rate of stretching at bottom

h₁(k) = ((-k + params.Nz) + 1) / params.Nz

# Linear near-surface generator
ζ₁(k) = 1 + (h₁(k) - 1) / refinement

# Bottom-intensified stretching function 
Σ₁(k) = (1 - exp(-stretching * h₁(k))) / (1 - exp(-stretching))

# Generating function
z_faces(k) = -params.Lz * (ζ₁(k) * Σ₁(k) - 1)

grid_base = RectilinearGrid(arch, topology = (Periodic, Bounded, Bounded),
                            size = (params.Nx, params.Ny, params.Nz),
                            x = (-params.Lx/2, +params.Lx/2),
                            y = (-params.y_offset, params.Ly-params.y_offset),
                            z = z_faces,
                            halo = (4,4,4),
                            )
@info grid_base
params = (; params..., Δz_min = minimum_zspacing(grid_base))
#---

#+++ Immersed boundary
include("bathymetry.jl")

#+++ Bathymetry visualization
if false
    bathymetry2(x, y, z) = seamount(x, y, z)

    xc = xnodes(grid_base, Center())
    yc = ynodes(grid_base, Center())
    zc = znodes(grid_base, Center())

    using GLMakie

    volume(xc, yc, zc, bathymetry2,
           isovalue = 1, isorange = 0.5,
           algorithm = :iso,
           axis=(type=Axis3, aspect=(params.Lx, params.Ly, 5params.Lz)))
    pause
end
#---

PCB = PartialCellBottom(seamount)

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

#+++ Boundary conditions for buoyancy
b∞(x, y, z, t, p) = p.N²∞ * z
b∞(x, z, t, p) = b∞(x, 0, z, t, p)

b_south = b_north = ValueBoundaryCondition(b∞, parameters = (; params.N²∞))
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
if CSM
    closure = SmagorinskyLilly(C=0.13, Pr=1)
elseif DSM
    closure = Smagorinsky(coefficient=DynamicCoefficient(averaging=LagrangianAveraging(), schedule=IterationInterval(5)))
else
    closure = AnisotropicMinimumDissipation()
end
#---

#+++ Model and ICs
@info "Creating model"
model = NonhydrostaticModel(grid = grid, timestepper = :RungeKutta3,
                            advection = WENO(grid=grid_base, order=5),
                            buoyancy = BuoyancyTracer(),
                            coriolis = FPlane(params.f_0),
                            tracers = :b,
                            closure = closure,
                            boundary_conditions = bcs,
                            forcing = (; u=Fᵤ),
                            hydrostatic_pressure_anomaly = CenterField(grid),
                            )
@info "" model
if has_cuda_gpu() run(`nvidia-smi -i $(ENV["CUDA_VISIBLE_DEVICES"])`) end

f_params = (; params.H, params.L, params.V∞, params.f₀, params.N²∞,)
set!(model, b=(x, y, z) -> b∞(x, y, z, 0, f_params), v=params.V∞)
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
                              + TimeStep() + "CFL = "*AdvectiveCFLNumber(with_prefix=false)
                              + "step dur = "*walltime_per_timestep)(simulation)
simulation.callbacks[:progress] = Callback(progress, IterationInterval(40))

wizard = TimeStepWizard(max_change=1.05, min_change=0.2, cfl=0.95, min_Δt=1e-4, max_Δt=1/√params.N²∞)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(4))

@info "" simulation
#---

#+++ Diagnostics
#+++ Check for checkpoints
if any(startswith("chk.$(params.simname)_iteration"), readdir("$rundir/data"))
    @warn "Checkpoint for $(params.simname) found. Assuming this is a pick-up simulation! Setting overwrite_existing=false."
    overwrite_existing = false
else
    @warn "No checkpoint for $(params.simname) found. Setting overwrite_existing=true."
    overwrite_existing = true
end
#---

include("$rundir/diagnostics.jl")
tick()
checkpointer = construct_outputs(simulation,
                                 simname = params.simname,
                                 rundir = rundir,
                                 params = params,
                                 overwrite_existing = overwrite_existing,
                                 interval_2d = 0.2*params.T_advective,
                                 interval_3d = 2.0*params.T_advective,
                                 interval_time_avg = 2*params.T_advective,
                                 write_xyz = true,
                                 write_xiz = false,
                                 write_xyi = true,
                                 write_iyz = true,
                                 write_ttt = false,
                                 write_tti = false,
                                 debug = false,
                                 )
tock()
#---

#+++ Run simulations and plot video afterwards
if has_cuda_gpu() run(`nvidia-smi -i $(ENV["CUDA_VISIBLE_DEVICES"])`) end
@info "Starting simulation"
run!(simulation, pickup=true)
#---

#+++ Plot video
include("$rundir/plot_video.jl")
#---
