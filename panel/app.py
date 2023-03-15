import holoviews as hv
import xarray as xr
import panel as pn

from holoviews.operation.datashader import rasterize

df = xr.open_dataset('outputs_1.nc', chunks={'time': 16}).persist()

renderer = hv.renderer('bokeh')

stream = hv.streams.Stream.define('time', time=0)()

def get_layout(time):
    img = rasterize(
        hv.Image(df.isel(time=time).temp),
        dynamic=False,
        width=64,
        height=64,
    )
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
        height=400,
        tools=['hover']
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
)

if __name__=="__main__":
    #from bokeh.resources import INLINE
    #app.save('visu', embed=True, resources=INLINE)
    pn.serve(app, start=True, show=False, port=40541)