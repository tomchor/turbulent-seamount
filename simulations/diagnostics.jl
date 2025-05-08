using Oceananigans.AbstractOperations: @at, ∂x, ∂y, ∂z
using Oceananigans.Units
using Oceananigans.Grids: Center, Face
import Oceananigans.TurbulenceClosures: viscosity, diffusivity

using Oceanostics: KineticEnergyDissipationRate,
                   ErtelPotentialVorticity, RossbyNumber, RichardsonNumber,
                   TracerVarianceDissipationRate, TurbulentKineticEnergy

viscosity(model)           = viscosity(model.closure, model.diffusivity_fields)
diffusivity(model, tracer) = diffusivity(model.closure, model.diffusivity_fields, tracer)


#+++ Methods/functions definitions
#include("$(@__DIR__)/grid_metrics.jl")
#---

#+++ Write to NCDataset
import NCDatasets as NCD
function write_to_ds(dsname, varname, data; coords=("x_caa", "y_aca", "z_aac"), dtype=Float64)
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

dbdx = @at CellCenter ∂x(b)
dbdy = @at CellCenter ∂y(b)
dbdz = @at CellCenter ∂z(b)

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

outputs_misc = Dict(pairs((; dbdx, dbdy, dbdz, ω_y,
                             Ri, Ro, PV,)))

outputs_diff = Dict(pairs((; ν, κ)))
#---

#+++ Define covariances
@info "Calculating covariances"
outputs_covs = Dict{Symbol, Any}(:uu => (@at CellCenter u*u),
                                 :vv => (@at CellCenter v*v),
                                 :ww => (@at CellCenter w*w),
                                 :uv => (@at CellCenter u*v),
                                 :uw => (@at CellCenter u*w),
                                 :vw => (@at CellCenter v*w),)
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

#+++ Define other energy budget terms
@info "Calculating energy budget terms"
wb = @at CellCenter w * b
outputs_budget = Dict{Symbol, Any}(:wb => wb,
                                   :Ek => TurbulentKineticEnergy(model, u, v, w),)
#---

#+++
@info "Defining volume averages"
outputs_vol_averages = Dict{Symbol, Any}(:∭⁵εₖdV => Integral(εₖ),
                                         :∭⁵εₚdV => Integral(εₚ),
                                         :∭⁵wbdV => Integral(wb),
                                         )
#outputs_vol_averages = Dict{Symbol, Any}(:∭⁵εₖdV => Integral(εₖ; condition = DistanceCondition(params_geometry, 5)),
#                                         :∭⁵εₚdV => Integral(εₚ; condition = DistanceCondition(params_geometry, 5)),
#                                         :∭⁵wbdV => Integral(wb; condition = DistanceCondition(params_geometry, 5)),
#                                         )
#---

#+++ Assemble the "full" outputs tuple
@info "Assemble diagnostics quantities"
outputs_full = merge(outputs_state_vars, outputs_dissip, outputs_misc, outputs_covs, outputs_grads, outputs_budget, outputs_diff)
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
                           write_xyz = false,
                           write_xiz = true,
                           write_xyi = false,
                           write_iyz = false,
                           write_ttt = false,
                           write_tti = false,
                           write_aaa = false,
                           write_chk = false,
                           debug = false,
                           )
    model = simulation.model

    #+++ Preamble and common keyword arguments
    k_half = ceil(Int, params.H / 2params.Δz_min) # Approximately half the seamount height
    kwargs = (overwrite_existing = overwrite_existing,
              deflatelevel = 5,
              global_attributes = params)
    #---

    #+++ xyz SNAPSHOTS
    if write_xyz
        @info "Setting up xyz writer"
        simulation.output_writers[:nc_xyz] = ow = NetCDFWriter(model, ScratchedField(outputs_full);
                                                               filename = "$rundir/data/xyz.$(simname).nc",
                                                               schedule = TimeInterval(interval_3d),
                                                               array_type = Array{Float64},
                                                               verbose = debug,
                                                               kwargs...
                                                               )
        @info "Starting to write grid metrics and deltas to xyz"
        laptimer()
        #add_grid_metrics_to!(ow)
        write_to_ds(ow.filepath, "altitude", interior(compute!(Field(altitude))), coords = ("x_caa", "y_aca", "z_aac"))
        write_to_ds(ow.filepath, "ΔxΔz", interior(compute!(Field(ΔxΔz))), coords = ("x_caa", "y_aca", "z_aac"))
        write_to_ds(ow.filepath, "bottom_height", Array(interior(maximum(compute!(Field(bottom_height)), dims=3)))[:,:,1], coords = ("x_caa", "y_aca",))
        @info "Finished writing grid metrics and deltas to xyz"
        laptimer()
    end
    #---

    #+++ xyi SNAPSHOTS
    if write_xyi
        @info "Setting up xyi writer"
        indices = (:, :, k_half)

        if write_aaa
            outputs_budget_integrated = Dict( Symbol(:∫∫∫, k, :dxdydz) => Integral(ScratchedField(v))  for (k, v) in outputs_budget )
            outputs_xyi = merge(outputs_xyi, outputs_budget_integrated)
        end

        simulation.output_writers[:nc_xyi] = ow = NetCDFWriter(model, outputs_full;
                                                               filename = "$rundir/data/xyi.$(simname).nc",
                                                               schedule = TimeInterval(interval_2d),
                                                               array_type = Array{Float64},
                                                               indices = indices,
                                                               verbose = debug,
                                                               kwargs...
                                                               )
        #add_grid_metrics_to!(ow, user_indices=indices)
    end
    #---

    #+++ xiz (low def) SNAPSHOTS
    if write_xiz
        @info "Setting up xiz writer"
        indices = (:, grid.Ny÷2, :)
        simulation.output_writers[:nc_xiz] = ow = NetCDFWriter(model, outputs_full;
                                                               filename = "$rundir/data/xiz.$(simname).nc",
                                                               schedule = TimeInterval(interval_2d),
                                                               array_type = Array{Float32},
                                                               indices = indices,
                                                               verbose = debug,
                                                               kwargs...
                                                               )

        #add_grid_metrics_to!(ow; user_indices=indices)
    end
    #---

    #+++ iyz (low def) SNAPSHOTS
    if write_iyz
        @info "Setting up iyz writer"
        indices = (grid.Nx÷2, :, :)
        simulation.output_writers[:nc_iyz] = ow = NetCDFWriter(model, outputs_full;
                                                               filename = "$rundir/data/iyz.$(simname).nc",
                                                               schedule = TimeInterval(interval_2d),
                                                               array_type = Array{Float32},
                                                               indices = indices,
                                                               verbose = debug,
                                                               kwargs...
                                                               )
        #add_grid_metrics_to!(ow, user_indices=indices)
    end
    #---

    #+++ ttt (Time averages)
    if write_ttt
        @info "Setting up ttt writer"
        outputs_ttt = merge(outputs_state_vars, outputs_dissip,)
        indices = (:, :, :)
        simulation.output_writers[:nc_ttt] = ow = NetCDFWriter(model, outputs_ttt;
                                                               filename = "$rundir/data/ttt.$(simname).nc",
                                                               schedule = AveragedTimeInterval(interval_time_avg, stride=10),
                                                               array_type = Array{Float64},
                                                               with_halos = false,
                                                               indices = indices,
                                                               verbose = true,
                                                               kwargs...
                                                               )
        #add_grid_metrics_to!(ow, user_indices=indices)
        write_to_ds(ow.filepath, "altitude", interior(compute!(Field(altitude, indices=indices))), coords = ("x_caa", "y_aca", "z_aac"))
        write_to_ds(ow.filepath, "bottom_height", Array(interior(maximum(compute!(Field(bottom_height)), dims=3)))[:,:,1], coords = ("x_caa", "y_aca",))
    end
    #---

    #+++ tti (Time averages)
    if write_tti
        @info "Setting up tti writer"
        outputs_tti = merge(outputs_full, outputs_vol_averages)
        indices = (:, :, k_half)
        simulation.output_writers[:nc_tti] = ow = NetCDFWriter(model, outputs_tti;
                                                               filename = "$rundir/data/tti.$(simname).nc",
                                                               schedule = AveragedTimeInterval(interval_time_avg, stride=10),
                                                               array_type = Array{Float64},
                                                               with_halos = false,
                                                               indices = indices,
                                                               verbose = debug,
                                                               kwargs...
                                                               )
        #add_grid_metrics_to!(ow, user_indices=indices)
    end
    #---

    #+++ Checkpointer
    @info "Setting up chk writer"
    if write_chk
        simulation.output_writers[:chk_writer] = checkpointer = Checkpointer(model;
                                                                             dir="$rundir/data/",
                                                                             prefix = "chk.$(simname)",
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
