using Oceananigans.TurbulenceClosures: _compute_AMD_viscosity!
using Oceananigans.Utils: launch!
import Oceananigans.TurbulenceClosures: compute_diffusivities!


function compute_diffusivities!(diffusivity_fields, closure::AnisotropicMinimumDissipation, model; parameters = :xyz)
    grid = model.grid
    arch = model.architecture
    velocities = model.velocities
    tracers = model.tracers
    buoyancy = model.buoyancy

    launch!(arch, grid, parameters, _compute_AMD_viscosity!, diffusivity_fields.νₑ, grid, closure, buoyancy, velocities, tracers)

    for (tracer_index, κₑ) in enumerate(diffusivity_fields.κₑ)
        @inbounds tracer = tracers[tracer_index]
        launch!(arch, grid, parameters, _compute_AMD_viscosity!, κₑ, grid, closure, buoyancy, velocities, tracers)
    end

    return nothing
end

