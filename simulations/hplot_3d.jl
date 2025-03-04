using Rasters
import NCDatasets
using Oceananigans: KernelFunctionOperation

simname = "tokara_α=0.2_Ro_h=1.25_Fr_h=0.5_res=2_closure=AMD_bounded=0"

@info "Reading NetCDF"
xyz = RasterStack("data/xyz.$simname.nc", name=(:PV,), lazy=true)

md = metadata(xyz)
params = (; (Symbol(k) => v for (k, v) in md)...)

#+++ Define bathymetry
@inline seamount(x, y, p) = p.H * exp(-((x - p.x₀)/p.L)^2 - ((y - p.y₀)/p.L)^2)
@inline seamount(x, y) = seamount(x, y, params)
@inline bathymetry(x, y, z) = z - seamount(x, y)
#---

@info "Slicing xyz to avoid border values"
xyz = xyz[:, 2:end-1, :, :]
xC = Array(dims(xyz, :xC))
yC = Array(dims(xyz, :yC))
zC = Array(dims(xyz, :zC))

PV_lim = 1.5 + params.Ro_h
PV_lims = (-PV_lim, +PV_lim)
bathymetry_array = [ bathymetry(x, y, z) < 5 ? 1 : 0 for x=xC, y=yC, z=zC ]

using GLMakie
fig = Figure(resolution = (1200, 900));
n = Observable(1)

PV = xyz.PV[Ti=Between(params.T_advective_spinup * params.T_advective, Inf)] ./ (md["N2_inf"] * md["f_0"])
PVₙ = @lift Array(PV)[:,:,:,$n]

colormap = to_colormap(:balance)
middle_chunk = ceil(Int, 1.5 * 128 / PV_lim) # Needs to be *at least* larger than 128 / PV_lim
colormap[128-middle_chunk:128+middle_chunk] .= RGBAf(0,0,0,0)

settings_axis3 = (aspect = (md["Lx"], md["Ly"], 4*md["Lz"]), azimuth = -0.80π, elevation = 0.15π,
                  perspectiveness=0.8, viewmode=:fitzoom, xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")

@info "Starting to plot things"
ax = Axis3(fig[1, 1]; settings_axis3...)
volume!(ax, xC, yC, zC, bathymetry_array, algorithm = :absorption, absorption=50f0, colormap = [RGBAf(0,0,0,0), :papayawhip], colorrange=(0, 1)) # turn on anti-aliasing

vol = volume!(ax, xC, yC, zC, PVₙ, algorithm = :absorption, absorption=20f0, colormap=colormap, colorrange=PV_lims)
Colorbar(fig, vol, bbox=ax.scene.px_area,
         label="PV / N²∞ f₀", height=25, width=Relative(0.35), vertical=false,
         alignmode = Outside(10), halign = 0.15, valign = 0.02)

#+++ Inset axis
inset_ax = Axis3(fig[1, 1]; width=Relative(0.4), height=Relative(0.4), halign=1, valign=1.1, zticks=[0, 40, 80], settings_axis3...)
volume!(inset_ax, xC, yC, zC, bathymetry_array, algorithm = :absorption, absorption=50f0, colormap = [RGBAf(0,0,0,0), :papayawhip], colorrange=(0, 1)) # turn on anti-aliasing

ps = [Point3f(0, -params.y_offset, 40)]
ns = [Vec3f(0, params.FWMH, 0)]
arrows!(inset_ax, ps, ns,
        fxaa = true, # turn on anti-aliasing
        color = :brown2, linewidth = 12, arrowsize = Vec3f(100, 20, 100))

text!(Point3f(-200, -100 + params.Ly/2, 40),
      text = "V∞ = 1 cm/s", rotation = 1.5π, align = (:left, :baseline),
      fontsize = 200, markerspace = :data, color = :brown2)
#---

#+++ Save a snapshot as png
n[] = length(dims(PV, :Ti))
resize_to_layout!(fig)
save(string(@__DIR__) * "/../figures/3d_PV_$simname.png", fig, px_per_unit=2);
#---

#+++ Define title with time
using Printf
using Oceananigans: prettytime
title = @lift "Frₕ = $(@sprintf "%.2g" params.Fr_h),    Roₕ = $(@sprintf "%.2g" params.Ro_h);    " *
              "Time = $(@sprintf "%s" prettytime(dims(PV, :Ti)[$n]))"
fig[0, 1] = Label(fig, title, fontsize=18, tellwidth=false, height=8)
#---

#+++ Record animation
n[] = 1
frames = 1:length(dims(PV, :Ti))
GLMakie.record(fig, string(@__DIR__) * "/../anims/3d_PV_$simname.mp4", frames, framerate=20) do frame
    @info "Plotting time step $frame"
    n[] = frame
end
#---
