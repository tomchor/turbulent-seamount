using Oceananigans.Fields: @compute
import Oceananigans.Grids: xnode, ynode, znode
using Adapt

xnode(i, grid, ℓx) = xnode(i, 1, 1, grid, ℓx, Center(), Center())
ynode(j, grid, ℓy) = ynode(1, j, 1, grid, Center(), ℓy, Center())
znode(k, grid, ℓz) = znode(1, 1, k, grid, Center(), Center(), ℓz)

#+++ Define it as z(x, y)
@inline seamount(x, y, p) = p.H * exp(-((x - p.x₀)/p.L)^2 - ((y - p.y₀)/p.L)^2)
@inline seamount(x, y) = seamount(x, y, params)
@inline seamount(i, j, k, grid, 𝓁x, 𝓁y, p) = seamount(xnode(i, grid, 𝓁x), ynode(j, grid, 𝓁y), p)
#---

xC = KernelFunctionOperation{Center, Center, Center}(xnode, grid_base, Center(), Center(), Center())
yC = KernelFunctionOperation{Center, Center, Center}(ynode, grid_base, Center(), Center(), Center())
zC = KernelFunctionOperation{Center, Center, Center}(znode, grid_base, Center(), Center(), Center())

using Oceananigans.Operators: xspacing, zspacing
ΔxΔz_kernel(i, j, k, grid, ℓx, ℓy, ℓz) = xspacing(i, j, k, grid, ℓx, ℓy, ℓz) * zspacing(i, j, k, grid, ℓx, ℓy, ℓz)
ΔxΔz = KernelFunctionOperation{Center, Center, Center}(ΔxΔz_kernel, grid_base, Center(), Center(), Center())

bottom_height = KernelFunctionOperation{Center, Center, Nothing}(seamount, grid_base, Center(), Center(), params_geometry)
