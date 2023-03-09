
from jax import numpy as jnp

def stream(dist_function, lattice):
    """_summary_

    Args:
        dist_function (_type_): _description_
        lattice (_type_): _description_

    Returns:
        _type_: _description_
    """
    for i in range(lattice.size):
        dist_function = dist_function.at[:, :, i].set(
            jnp.roll(
                jnp.roll(
                    dist_function[:, :, i],
                    lattice.coords[0, i],
                    axis=0,
                ),
                lattice.coords[1, i],
                axis=1,
            )
        )
    return dist_function