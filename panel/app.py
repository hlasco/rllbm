import holoviews as hv

import xarray as xr
import panel as pn


df = xr.open_dataset('../outputs_1.nc', chunks={'time':64})

renderer = hv.renderer('bokeh')

stream = hv.streams.Stream.define('time', time=0)()

def get_image(time):
    img = hv.Image(df.temp.isel(time=time))
    img.opts()
    
    layout = (img + img.sample(ny=32)).cols(1)
    return layout

dmap = hv.DynamicMap(get_image, streams=[stream]).opts(
    hv.opts.Image(
        cmap="RdBu_r",
        clim=(-0.5, 0.5),
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
        ylim=(-0.5, 0.5),
        ylabel="Bottom Temperature"),
)

slider = pn.widgets.IntSlider(
    start=0,
    end=(len(df.time)-1),
    value=0,
    step=1,
    name="Snapshot"
)

# Create a slider and play buttons
def animate_update():
    time = (slider.value + 1) % (len(df.time)-1)
    slider.value = time

def slider_update(event):
    stream.event(time=event.new)
    
def animate(event):
    if button.name == '► Play':
        button.name = '❚❚ Pause'
        callback.start()
    else:
        button.name = '► Play'
        callback.stop()
    
slider.param.watch(slider_update, 'value')

button = pn.widgets.Button(name='► Play', width=60, align='end')
button.on_click(animate)
callback = pn.state.add_periodic_callback(animate_update, 50, start=False)


app = pn.Column(
    dmap,
    pn.Row(slider, button),
).servable(title='HoloViews App')

if __name__=="__main__":
    pn.serve(app, start=True, show=True)