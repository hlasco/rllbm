import holoviews as hv
import xarray as xr
import panel as pn

from holoviews.operation.datashader import rasterize

df = xr.open_dataset("outputs.nc")

hv.extension("bokeh")
stream = hv.streams.Stream.define("time", time=0)()


def temp_image(time):
    img = rasterize(
        hv.Image(
            (df.x, df.y, df.isel(time=time).temp.T),
        ),
        width=256,
        height=256,
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
        colorbar_opts={"title": "Fluid Temperature"},
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
        xticks=(-0.5, 0, 0.5),
        xlabel="Wall Temperature",
    ),
)

player = pn.widgets.Player(
    name="Player",
    start=0,
    end=(len(df.time) - 1)//2,
    value=0,
    loop_policy="once",
    width=550,
    interval=100,
)


@pn.depends(time=player.param.value)
def animate(time):
    stream.event(time=time*2)


layout = curve + img

app = pn.Column(
    layout,
    player,
    animate,
)

if __name__ == "__main__":
    pn.serve(app, start=True, show=True)
