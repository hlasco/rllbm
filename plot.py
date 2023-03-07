import plotly.express as px
import xarray as xr
# Load xarray from dataset included in the xarray tutorial
ds = xr.open_dataset('outputs_1.nc')
print(ds)
fig = px.imshow(ds.temp.T, animation_frame='time', zmin=-.7, zmax=0.7, color_continuous_scale='RdBu_r', origin='lower', aspect='equal')
fig.show()