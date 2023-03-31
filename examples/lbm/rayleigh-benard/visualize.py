import holoviews as hv
import xarray as xr
import panel as pn

from holoviews.operation.datashader import rasterize

df = xr.open_dataset("outputs.nc")

hv.extension("bokeh")
stream = hv.streams.Stream.define("time", time=0)()

xmin, xmax = df.x.values.min(), df.x.values.max()
ymin, ymax = df.y.values.min(), df.y.values.max()
aspect = (ymax - ymin) / (xmax - xmin)
aspect = [min(1/aspect, 1), min(aspect, 1)]

def temp_image(time):
    img = rasterize(
        hv.Image(
            (df.x, df.y, df.isel(time=time).temp.T),
        ),
        #width=int(256 * aspect[0]),
        #height=int(256 * aspect[1]),
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
        clim=(-0.05, 0.05),
        xlim=(xmin, xmax),
        ylim=(ymin, ymax),
        ylabel="",
        yticks=0,
        colorbar=True,
        colorbar_position="right",
        colorbar_opts={"title": "Fluid Temperature"},
        frame_width = int(400 * aspect[0]),
        frame_height = int(400 * aspect[1]),
        tools=["hover"],
        framewise=True,
    ),
)

curve = hv.DynamicMap(temp_curve, streams=[stream]).opts(
    hv.opts.Curve(
        framewise=True,
        frame_width = int(100 * aspect[0]),
        frame_height = int(400 * aspect[1]),
        ylim=(ymin, ymax),
        xlim=(-0.55, 0.55),
        xticks=(-0.5, 0, 0.5),
        xlabel="Wall Temperature",
    ),
)

player = pn.widgets.Player(
    name="Player",
    start=0,
    end=(len(df.time) - 1),
    value=0,
    loop_policy="once",
    width=int(500 * aspect[0]),
    interval=100,
)


@pn.depends(time=player.param.value)
def animate(time):
    stream.event(time=time)


layout = curve + img

app = pn.Column(
    layout,
    pn.Row(pn.layout.HSpacer(), player, pn.layout.HSpacer()),
    animate,
)

if __name__ == "__main__":
    pn.serve(app, start=True, show=False, port=48070)
