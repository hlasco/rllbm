
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
        dist_function = dist_function.at[..., i].set(
            jnp.roll(
                a = dist_function[..., i],
                shift = lattice.coords[:,i],
                axis = [k for k in range(lattice.dim)]
            )
        )
    return dist_function