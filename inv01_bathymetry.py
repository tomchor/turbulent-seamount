import xarray as xr
from matplotlib import pyplot as plt
from cmocean import cm

ds = xr.load_dataset("gebco_2024_n33.0_s28.0_w128.0_e134.0.nc")
zoom_in = ds.sel(lat=slice(30, 30.10, 4), lon=slice(129.6, 130.2))

ds.elevation.plot(cmap=cm.topo)
plt.figure()

zoom_in.elevation.plot(hue="lat", ylim=(None, 0))
plt.figure()

x = (zoom_in.lon - zoom_in.lon[0]) * 96 # km for lat=30N
zoom_in_km = zoom_in.assign_coords(lon=x).rename(lon="x")
zoom_in_km.elevation.plot.line(x="x", ylim=(None, 0))
