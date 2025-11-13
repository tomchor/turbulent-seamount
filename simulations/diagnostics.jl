using Oceananigans.AbstractOperations: @at, ∂x, ∂y, ∂z
using Oceananigans.Units
using Oceananigans.Grids: Center, Face
import Oceananigans.TurbulenceClosures: viscosity, diffusivity

using Oceanostics: KineticEnergyDissipationRate, KineticEnergyForcing,
                   ErtelPotentialVorticity, DirectionalErtelPotentialVorticity, RossbyNumber, RichardsonNumber,
                   TracerVarianceDissipationRate

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
ω_x = @at CellCenter (∂y(w) - ∂z(v))

if model.closure isa Nothing
    εₖ = @at CellCenter CenterField(grid)
    εₚ = @at CellCenter CenterField(grid)

    κ = CenterField(grid)

else
    εₖ = @at CellCenter KineticEnergyDissipationRate(model)
    εₚ = @at CellCenter TracerVarianceDissipationRate(model, :b)/(2params.N2_inf)

    κ = diffusivity(model, Val(:b))
end


using Oceananigans.Fields: FunctionField
mask_top_u = Oceananigans.Fields.FunctionField{Face, Center, Center}(mask_top, grid) |> Field
mask_top_v = Oceananigans.Fields.FunctionField{Center, Face, Center}(mask_top, grid) |> Field
mask_top_w = Oceananigans.Fields.FunctionField{Center, Center, Face}(mask_top, grid) |> Field

const σ = params.sponge_damping_rate
const U∞ = params.U∞
u_sponge_field = -σ * (u - U∞) * mask_top_u
v_sponge_field = -σ * v * mask_top_v
w_sponge_field = -σ * w * mask_top_w

εₛ_u = @at CellCenter u * u_sponge_field |> Field
εₛ_aux = @at CellCenter (εₛ_u + v * v_sponge_field) |> Field
εₛ = @at CellCenter (εₛ_aux + w * w_sponge_field) |> Field

Ri = @at CellCenter RichardsonNumber(model, u, v, w, b)
Ro = @at CellCenter RossbyNumber(model)
PV = @at CellCenter ErtelPotentialVorticity(model, u, v, w, b, model.coriolis)
PV_z = @at CellCenter DirectionalErtelPotentialVorticity(model, (0, 0, 1))

outputs_dissip = Dict(pairs((; εₖ, εₚ, κ, εₛ)))
outputs_misc = Dict(pairs((; ω_x, Ri, Ro, PV, PV_z)))
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
# Define conditions to avoid unresolved bottom, sponge layer, and couple of points closest to the eastern open boundary
dc5  = DistanceCondition(from_bottom=5meters , from_top=params.h_sponge, from_east=2minimum_xspacing(grid))
dc10 = DistanceCondition(from_bottom=10meters, from_top=params.h_sponge, from_east=2minimum_xspacing(grid))

outputs_vol_integrals = Dict{Symbol, Any}(:∭εₛdV   => Integral(εₛ),
                                          :∭⁵εₖdV  => Integral(εₖ; condition = dc5),
                                          :∭⁵εₚdV  => Integral(εₚ; condition = dc5),
                                          :∭¹⁰εₖdV => Integral(εₖ; condition = dc10),
                                          :∭¹⁰εₚdV => Integral(εₚ; condition = dc10),
                                          )

outputs_x_integrals = Dict{Symbol, Any}(:∫εₛdy   => Integral(εₛ; dims=2),
                                        :∫⁵εₖdy   => Integral(εₖ; condition = dc5, dims=2),
                                        :∫⁵εₚdy   => Integral(εₚ; condition = dc5, dims=2),
                                        :∫¹⁰εₖdy  => Integral(εₖ; condition = dc10, dims=2),
                                        :∫¹⁰εₚdy  => Integral(εₚ; condition = dc10, dims=2),
                                        )

dcf5  = Field(KernelFunctionOperation{Center, Center, Center}(dc5,  grid, nothing)) |> compute!
dcf10 = Field(KernelFunctionOperation{Center, Center, Center}(dc10, grid, nothing)) |> compute!
#---

#+++ Assemble the "full" outputs tuple
@info "Assemble diagnostics quantities"
outputs_full = merge(outputs_state_vars, outputs_dissip, outputs_misc, outputs_covs, outputs_grads)
#---
#---

#+++ Construct outputs into simulation
function construct_outputs(simulation;
                           simname = "TEST",
                           rundir = @__DIR__,
                           params = params,
                           overwrite_existing = overwrite_existing,
                           interval_2d = 0.2*params.T_adv,
                           interval_3d = params.T_adv,
                           interval_time_avg = 20*params.T_adv,
                           write_xyzi = false,
                           write_xizi = false,
                           write_xyii = true,
                           write_iyzi = false,
                           write_xyza = false,
                           write_xyia = false,
                           write_aaai = false,
                           write_ckpt = false,
                           debug = false,
                           )
    model = simulation.model
    grid = model.grid

    #+++ Preamble and common keyword arguments
    k_xy_slice = ceil(Int, params.H / 3params.Δz_min) # Approximately 1/3 the seamount height
    kwargs = (overwrite_existing = overwrite_existing,
              deflatelevel = 0, # Speeds up reading and writing. Should make a big difference in offline averaging
              global_attributes = params)
    #---

    #+++ xyzi SNAPSHOTS
    if write_xyzi
        @info "Setting up xyzi writer"
        outputs_xyzi = merge(ScratchedField(outputs_full), outputs_x_integrals)
        simulation.output_writers[:nc_xyzi] = ow = @CUDAstats NetCDFWriter(model, outputs_xyzi;
                                                                           filename = "$rundir/data/xyzi.$(simname).nc",
                                                                           schedule = TimeInterval(interval_3d),
                                                                           array_type = Array{eltype(grid)},
                                                                           verbose = debug,
                                                                           kwargs...
                                                                           )
        write_to_ds(ow.filepath, "altitude", interior(compute!(Field(altitude))), coords = ("x_caa", "y_aca", "z_aac"))
        write_to_ds(ow.filepath, "distance_condition_5meters",  interior(dcf5),  coords = ("x_caa", "y_aca", "z_aac"))
        write_to_ds(ow.filepath, "distance_condition_10meters", interior(dcf10), coords = ("x_caa", "y_aca", "z_aac"))
    end
    #---

    #+++ xyii SNAPSHOTS
    if write_xyii
        @info "Setting up xyii writer"
        indices = (:, :, k_xy_slice)
        simulation.output_writers[:nc_xyii] = @CUDAstats NetCDFWriter(model, outputs_full;
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
        simulation.output_writers[:nc_xizi] = @CUDAstats NetCDFWriter(model, outputs_full;
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
        simulation.output_writers[:nc_iyzi] = @CUDAstats NetCDFWriter(model, outputs_full;
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
        simulation.output_writers[:nc_xyza] = ow = @CUDAstats NetCDFWriter(model, outputs_xyza;
                                                                           filename = "$rundir/data/xyza.$(simname).nc",
                                                                           schedule = AveragedTimeInterval(interval_time_avg, stride=10),
                                                                           array_type = Array{eltype(grid)},
                                                                           verbose = true,
                                                                           kwargs...
                                                                           )
        write_to_ds(ow.filepath, "altitude", interior(compute!(Field(altitude))), coords = ("x_caa", "y_aca", "z_aac"))
        write_to_ds(ow.filepath, "distance_condition_5meters",  interior(dcf5),  coords = ("x_caa", "y_aca", "z_aac"))
        write_to_ds(ow.filepath, "distance_condition_10meters", interior(dcf10), coords = ("x_caa", "y_aca", "z_aac"))
    end
    #---

    #+++ xyia (Time averages)
    if write_xyia
        @info "Setting up xyia writer"
        outputs_xyia = merge(outputs_full, outputs_vol_integrals)
        indices = (:, :, k_xy_slice)
        simulation.output_writers[:nc_xyia] = @CUDAstats NetCDFWriter(model, outputs_xyia;
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

    #+++ aaai (Time averages)
    if write_aaai
        @info "Setting up aaai writer"
        simulation.output_writers[:nc_aaai] = ow = @CUDAstats NetCDFWriter(model, outputs_vol_integrals;
                                                                           filename = "$rundir/data/aaai.$(simname).nc",
                                                                           schedule = TimeInterval(interval_2d),
                                                                           array_type = Array{eltype(grid)},
                                                                           verbose = false,
                                                                           kwargs...
                                                                           )
    end
    #---
end
#---
