import holoviews as hv
import xarray as xr
import panel as pn

from holoviews.operation.datashader import rasterize

df = xr.open_dataset("outputs_1.nc")

hv.extension("bokeh")
stream = hv.streams.Stream.define("time", time=0)()


def temp_image(time):
    img = rasterize(
        hv.Image(
            (df.x, df.y, df.isel(time=time).temp.T),
        ),
        dynamic=False,
    )
    return img


def temp_curve(time):
    curve = hv.Curve(
        (
            df.isel(time=time, nx=0).temp,
            df.y,
        ),
        "T_0",
        "y",
    )
    return curve


def get_tracers(time):
    tracers_data = df.tracers.isel(time=slice(max(0, time - 15), time + 1))
    tracers = [hv.Points((tracers_data[k],)) for k in tracers_data.tracer_id.data]
    return hv.Overlay(tracers)


img = hv.DynamicMap(temp_image, streams=[stream]).opts(
    hv.opts.Image(
        cmap="RdBu_r",
        symmetric=True,
        clim=(-0.1, 0.1),
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        ylabel="",
        yticks=0,
        colorbar=True,
        colorbar_position="top",
        width=400,
        height=400,
        aspect="equal",
        tools=["hover"],
        framewise=False,
    ),
)

curve = hv.DynamicMap(temp_curve, streams=[stream]).opts(
    hv.opts.Curve(
        framewise=True,
        width=150,
        height=400,
        ylim=(0.0, 1.0),
        xlim=(-0.55, 0.55),
        xlabel="Temperature",
    ),
)

tracers = hv.DynamicMap(get_tracers, streams=[stream]).opts(
    hv.opts.Points(marker="o", size=5, alpha=0.5),
)

player = pn.widgets.Player(
    name="Player",
    start=0,
    end=(len(df.time) - 1),
    value=0,
    loop_policy="once",
    width=550,
    interval=100,
)


@pn.depends(time=player.param.value)
def animate(time):
    stream.event(time=time)


layout = curve + img * tracers

app = pn.Column(
    layout,
    player,
    animate,
)

if __name__ == "__main__":
    pn.serve(app, start=True, show=False, port=40541)
