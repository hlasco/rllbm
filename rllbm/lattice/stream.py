
from jax import numpy as jnp
from jax.typing import ArrayLike
from typing import Tuple, overload
from rllbm.lattice import Lattice

@overload
def stream(lattice: Lattice, mask: ArrayLike[bool]):
    """_summary_

    Args:
        dist_function (_type_): _description_
        lattice (_type_): _description_

    Returns:
        _type_: _description_
    """
    for i in range(lattice.Q):
        lattice.df = lattice.df.at[..., i].set(
            jnp.where(
                mask,
                jnp.roll(
                    a = lattice.df[..., i],
                    shift = lattice.e[:,i],
                    axis = [k for k in range(lattice.D)]
                ),
                lattice.df[...,i]
            )
        )
    return lattice

@overload
def stream(lattice_list: Tuple[Lattice], mask_list: Tuple[Lattice]) -> Tuple[Lattice]:
    assert len(lattice_list) == len(mask_list)
    n = len(lattice_list)
    for i in range(n):
        lattice_list[i] = stream(lattice_list[i], mask_list[i])
    
    return lattice_list