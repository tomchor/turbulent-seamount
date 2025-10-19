import xarray as xr

sars = xr.open_dataset("GMRT/DEM_20251018DEM_sars.nc")
bala = xr.open_dataset("GMRT/GMRTv4_3_1_20250502topo_balanus.nc")

sars = sars.assign_coords(lon = sars.lon - sars.lon.mean(), lat = sars.lat - sars.lat.mean())
bala = bala.assign_coords(lon = bala.lon - bala.lon.mean(), lat = bala.lat - bala.lat.mean())

sars.sel(lat=0, method="nearest").z.plot.scatter(x="lon", edgecolor="none", label="sars")
bala.sel(lat=0, method="nearest").z.plot.scatter(x="lon", edgecolor="none", label="balanus")
