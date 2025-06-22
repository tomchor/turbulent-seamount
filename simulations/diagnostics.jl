using Oceananigans.AbstractOperations: @at, ∂x, ∂y, ∂z
using Oceananigans.Units
using Oceananigans.Grids: Center, Face
import Oceananigans.TurbulenceClosures: viscosity, diffusivity
using CUDA  # Add CUDA import
#using MacroTools  # Add MacroTools for macro writing

using Oceanostics: KineticEnergyDissipationRate,
                   ErtelPotentialVorticity, RossbyNumber, RichardsonNumber,
                   TracerVarianceDissipationRate, TurbulentKineticEnergy

viscosity(model)           = viscosity(model.closure, model.diffusivity_fields)
diffusivity(model, tracer) = diffusivity(model.closure, model.diffusivity_fields, tracer)


#+++ Write to NCDataset
import NCDatasets as NCD
function write_to_ds(dsname, varname, data; coords=("x_caa", "y_aca", "z_aac"), dtype=eltype(grid))
    ds = NCD.NCDataset(dsname, "a")
    if varname ∉ keys(ds)
        newvar = NCD.defVar(ds, varname, dtype, coords)
        if length(size(data)) == 3
            newvar[:,:,:] = Array(data)
        elseif length(size(data)) == 2
            newvar[:,:] = Array(data)
        elseif length(size(data)) == 1
            newvar[:] = Array(data)
        else
            newvar = data
        end
    end
    NCD.close(ds)
end
#---

#+++ Define Fields
using Oceananigans.AbstractOperations: AbstractOperation
import Oceananigans.Fields: Field

ccc_scratch = Field{Center, Center, Center}(model.grid)
ScratchedField(op::AbstractOperation{Center, Center, Center}) = Field(op, data=ccc_scratch.data)

ScratchedField(f::Field) = f
ScratchedField(d::Dict) = Dict( k => ScratchedField(v) for (k, v) in d )
ScratchedField(n::Number) = ScratchedField(n * CenterField(grid))
#---

#+++ Unpack model variables
CellCenter = (Center, Center, Center) # Output everything on cell centers to make life easier
u, v, w = model.velocities
b = model.tracers.b

outputs_vels = Dict{Any, Any}(:u => (@at CellCenter u),
                              :v => (@at CellCenter v),
                              :w => (@at CellCenter w),)
outputs_state_vars = merge(outputs_vels, Dict{Any, Any}(:b => b))
#---

#+++ CREATE SNAPSHOT OUTPUTS
#+++ Start calculation of snapshot variables
@info "Calculating misc diagnostics"

@compute altitude = Field(KernelFunctionOperation{Center, Center, Center}(z_distance_from_seamount_boundary_ccc, grid))
ω_y = @at CellCenter (∂z(u) - ∂x(w))

if model.closure isa Nothing
    εₖ = @at CellCenter CenterField(grid)
    εₚ = @at CellCenter CenterField(grid)

    ν = CenterField(grid)
    κ = CenterField(grid)

else
    εₖ = @at CellCenter KineticEnergyDissipationRate(model)
    εₚ = @at CellCenter TracerVarianceDissipationRate(model, :b)/(2params.N2_inf)

    ν = viscosity(model)
    κ = diffusivity(model, Val(:b))
end

Ri = @at CellCenter RichardsonNumber(model, u, v, w, b)
Ro = @at CellCenter RossbyNumber(model)
PV = @at CellCenter ErtelPotentialVorticity(model, u, v, w, b, model.coriolis)

outputs_dissip = Dict(pairs((; εₖ, εₚ, κ)))

outputs_misc = Dict(pairs((; ω_y, Ri, Ro, PV,)))

outputs_diff = Dict(pairs((; ν, κ)))
#---

#+++ Define covariances
@info "Calculating covariances"
outputs_covs = Dict{Symbol, Any}(:uu => (@at CellCenter u*u),
                                 :vv => (@at CellCenter v*v),
                                 :ww => (@at CellCenter w*w),
                                 :uv => (@at CellCenter u*v),
                                 :uw => (@at CellCenter u*w),
                                 :vw => (@at CellCenter v*w),
                                 :wb => (@at CellCenter w*b))
#---

#+++ Define velocity gradient tensor
@info "Calculating velocity gradient tensor"
outputs_grads = Dict{Symbol, Any}(:∂u∂x => (@at CellCenter ∂x(u)),
                                  :∂v∂x => (@at CellCenter ∂x(v)),
                                  :∂w∂x => (@at CellCenter ∂x(w)),
                                  :∂u∂y => (@at CellCenter ∂y(u)),
                                  :∂v∂y => (@at CellCenter ∂y(v)),
                                  :∂w∂y => (@at CellCenter ∂y(w)),
                                  :∂u∂z => (@at CellCenter ∂z(u)),
                                  :∂v∂z => (@at CellCenter ∂z(v)),
                                  :∂w∂z => (@at CellCenter ∂z(w)),)
#---

#+++ Volume averages
@info "Defining volume averages"
# Define conditions to avoid unresolved bottom, sponge layer, and couple of points closest to the
# open boundary
dc5  = DistanceCondition(from_bottom=5meters , from_top=params.h_sponge, from_north=2minimum_yspacing(grid))
dc10 = DistanceCondition(from_bottom=10meters, from_top=params.h_sponge, from_north=2minimum_yspacing(grid))
dc20 = DistanceCondition(from_bottom=20meters, from_top=params.h_sponge, from_north=2minimum_yspacing(grid))

outputs_vol_averages = Dict{Symbol, Any}(:∭⁵εₖdV  => Integral(εₖ; condition = dc5),
                                         :∭⁵εₚdV  => Integral(εₚ; condition = dc5),
                                         :∭¹⁰εₖdV => Integral(εₖ; condition = dc10),
                                         :∭¹⁰εₚdV => Integral(εₚ; condition = dc10),
                                         :∭²⁰εₖdV => Integral(εₖ; condition = dc20),
                                         :∭²⁰εₚdV => Integral(εₚ; condition = dc20),
                                         )

dcf5  = Field(KernelFunctionOperation{Center, Center, Center}(dc5,  grid, nothing)) |> compute!
dcf10 = Field(KernelFunctionOperation{Center, Center, Center}(dc10, grid, nothing)) |> compute!
dcf20 = Field(KernelFunctionOperation{Center, Center, Center}(dc20, grid, nothing)) |> compute!
#---

#+++ Assemble the "full" outputs tuple
@info "Assemble diagnostics quantities"
outputs_full = merge(outputs_state_vars, outputs_dissip, outputs_misc, outputs_covs, outputs_grads, outputs_diff)
#---
#---

#+++ Construct outputs into simulation
function construct_outputs(simulation; 
                           simname = "TEST",
                           rundir = @__DIR__,
                           params = params,
                           overwrite_existing = overwrite_existing,
                           interval_2d = 0.2*params.T_advective,
                           interval_3d = params.T_advective,
                           interval_time_avg = 20*params.T_advective,
                           write_xyzi = false,
                           write_xizi = true,
                           write_xyii = false,
                           write_iyzi = false,
                           write_xyza = false,
                           write_xyia = false,
                           write_ckpt = false,
                           debug = false,
                           )
    model = simulation.model
    grid = model.grid

    #+++ Preamble and common keyword arguments
    k_half = ceil(Int, params.H / 2params.Δz_min) # Approximately half the seamount height
    kwargs = (overwrite_existing = overwrite_existing,
              deflatelevel = 5,
              global_attributes = params)
    #---

    #+++ xyzi SNAPSHOTS
    if write_xyzi
        @info "Setting up xyzi writer"
        simulation.output_writers[:nc_xyzi] = ow = @measure_memory NetCDFWriter(model, ScratchedField(outputs_full);
                                                                                filename = "$rundir/data/xyzi.$(simname).nc",
                                                                                schedule = TimeInterval(interval_3d),
                                                                                array_type = Array{eltype(grid)},
                                                                                verbose = debug,
                                                                                kwargs...
                                                                                )
        write_to_ds(ow.filepath, "altitude", interior(compute!(Field(altitude))), coords = ("x_caa", "y_aca", "z_aac"))
        write_to_ds(ow.filepath, "distance_condition_5meters",  interior(dcf5),  coords = ("x_caa", "y_aca", "z_aac"))
        write_to_ds(ow.filepath, "distance_condition_10meters", interior(dcf10), coords = ("x_caa", "y_aca", "z_aac"))
        write_to_ds(ow.filepath, "distance_condition_20meters", interior(dcf20), coords = ("x_caa", "y_aca", "z_aac"))
    end
    #---

    #+++ xyii SNAPSHOTS
    if write_xyii
        @info "Setting up xyii writer"
        indices = (:, :, k_half)
        simulation.output_writers[:nc_xyii] = @measure_memory NetCDFWriter(model, outputs_full;
                        filename = "$rundir/data/xyii.$(simname).nc",
                        schedule = TimeInterval(interval_2d),
                        array_type = Array{eltype(grid)},
                        indices = indices,
                        verbose = debug,
                        kwargs...
                        )
    end
    #---

    #+++ xizi SNAPSHOTS
    if write_xizi
        @info "Setting up xizi writer"
        indices = (:, grid.Ny÷2, :)
        simulation.output_writers[:nc_xizi] = @measure_memory NetCDFWriter(model, outputs_full;
                        filename = "$rundir/data/xizi.$(simname).nc",
                        schedule = TimeInterval(interval_2d),
                        array_type = Array{eltype(grid)},
                        indices = indices,
                        verbose = debug,
                        kwargs...
                        )
    end
    #---

    #+++ iyzi SNAPSHOTS
    if write_iyzi
        @info "Setting up iyzi writer"
        indices = (grid.Nx÷2, :, :)
        simulation.output_writers[:nc_iyzi] = @measure_memory NetCDFWriter(model, outputs_full;
                        filename = "$rundir/data/iyzi.$(simname).nc",
                        schedule = TimeInterval(interval_2d),
                        array_type = Array{eltype(grid)},
                        indices = indices,
                        verbose = debug,
                        kwargs...
                        )
    end
    #---

    #+++ xyza (Time averages)
    if write_xyza
        @info "Setting up xyza writer"
        outputs_xyza = merge(outputs_state_vars, outputs_dissip, outputs_covs)
        simulation.output_writers[:nc_xyza] = ow = @measure_memory NetCDFWriter(model, outputs_xyza;
                                                                                filename = "$rundir/data/xyza.$(simname).nc",
                                                                                schedule = AveragedTimeInterval(interval_time_avg, stride=10),
                                                                                array_type = Array{eltype(grid)},
                                                                                verbose = true,
                                                                                kwargs...
                                                                                )
        write_to_ds(ow.filepath, "altitude", interior(compute!(Field(altitude))), coords = ("x_caa", "y_aca", "z_aac"))
        write_to_ds(ow.filepath, "distance_condition_5meters",  interior(dcf5),  coords = ("x_caa", "y_aca", "z_aac"))
        write_to_ds(ow.filepath, "distance_condition_10meters", interior(dcf10), coords = ("x_caa", "y_aca", "z_aac"))
        write_to_ds(ow.filepath, "distance_condition_20meters", interior(dcf20), coords = ("x_caa", "y_aca", "z_aac"))
    end
    #---

    #+++ xyia (Time averages)
    if write_xyia
        @info "Setting up xyia writer"
        outputs_xyia = merge(outputs_full, outputs_vol_averages)
        indices = (:, :, k_half)
        simulation.output_writers[:nc_xyia] = @measure_memory NetCDFWriter(model, outputs_xyia;
                        filename = "$rundir/data/xyia.$(simname).nc",
                        schedule = AveragedTimeInterval(interval_time_avg, stride=10),
                        array_type = Array{eltype(grid)},
                        with_halos = false,
                        indices = indices,
                        verbose = debug,
                        kwargs...
                        )
    end
    #---

    #+++ Checkpointer
    @info "Setting up ckpt writer"
    if write_ckpt
        simulation.output_writers[:ckpt_writer] = checkpointer = @measure_memory Checkpointer(model;
                                                                             dir="$rundir/data/",
                                                                             prefix = "ckpt.$(simname)",
                                                                             schedule = TimeInterval(interval_time_avg),
                                                                             overwrite_existing = true,
                                                                             cleanup = true,
                                                                             verbose = debug,
                                                                             )
        return checkpointer
    else
        return nothing
    end
    #---
end
#---
