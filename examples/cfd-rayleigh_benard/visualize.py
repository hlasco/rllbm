
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import LogFormatter

def fig_constructor(sim):
    fig, ax = plt.subplots(figsize=(6, 4))
    
    temperature = sim.fluid_state.T[:, :, 0].T

    x = sim.x
    y = sim.y
    vmax = 0.5

    img = plt.imshow(
        temperature,
        extent=[x.min(), x.max(), y.min(), y.max()],
        cmap="RdBu_r",
        norm=SymLogNorm(0.1*vmax, vmin=-vmax, vmax=vmax),
        origin="lower",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    formatter = LogFormatter(10, labelOnlyBase=False)

    cbar = fig.colorbar(img, ax=ax, format=formatter )
    cbar.set_label("Fluid Temperature")

    fig.tight_layout(pad=2)
    fig.canvas.draw()
    fig_data = {
        "figure": fig,
        "image": img,
        "axes": ax,
    }
    return fig_data

def fig_updater(fig_data, sim):
    temperature = sim.fluid_state.T[:, :, 0].T
    fig_data["image"].set_data(temperature)
    return fig_data

