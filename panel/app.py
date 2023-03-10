import holoviews as hv
import xarray as xr
import panel as pn
import numpy as np

import datashader as ds
from datashader import transfer_functions as tf

from streamline import streamlines

df = xr.open_dataset('outputs_1.nc', chunks={'time': 16}).persist()

renderer = hv.renderer('bokeh')

stream = hv.streams.Stream.define('time', time=0)()

def get_layout(time):
    cvs = ds.Canvas(plot_width=196, plot_height=196)
    img = hv.Image(cvs.raster(df.isel(time=time).temp.T))
    curve = hv.Curve(df.isel(time=time, ny=0).temp)
    layout = (img + curve).cols(1)
    return layout

dmap = hv.DynamicMap(get_layout, streams=[stream]).opts(
    hv.opts.Image(
        cmap="RdBu_r",
        clim=(-0.04, 0.04),
        xlabel="",
        xticks=0,
        colorbar=True,
        colorbar_position='top',
        width=400,
        height=400
    ),
    hv.opts.Curve(
        framewise=True,
        width=400,
        height=150,
        ylim=(-0.55, 0.55),
        ylabel="Bottom Temperature"),
)

player = pn.widgets.Player(
    name='Player',
    start=0,
    end=(len(df.time)-1),
    value=0,
    loop_policy='once',
    width=400,
    interval=200,
)

@pn.depends(time=player.param.value)
def animate(time):
    stream.event(time=time)

app = pn.Column(
    dmap,
    player,
    animate,
).servable(title='HoloViews App')

if __name__=="__main__":
    pn.serve(app, start=True, show=False, port=40541)