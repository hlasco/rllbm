from typing import Tuple, Union, List
from functools import partial

from jax import numpy as jnp
from jax.typing import ArrayLike
from jax import Array, jit

from rllbm.lbm.lattice import Lattice, CoupledLattices

__all__ = ["stream"]


@partial(jit, static_argnums=(0))
def _stream(lattice: Lattice, dist_function: ArrayLike, mask: ArrayLike) -> Array:
    """This function takes in a distribution function, and a mask, and streams the
    distribution function according to the mask.
    The mask allows to stream the distribution function only on the fluid nodes, and not
    on the boundary nodes.

    Args:
        lattice (Lattice): The lattice on which the streaming is performed.
        dist_function (ArrayLike): The distribution function to be streamed.
        mask (ArrayLike): The mask where the streaming should be performed.

    Returns:
        Array: The streamed distribution function.
    """
    for i in range(lattice.Q):
        dist_function = dist_function.at[..., i].set(
            jnp.where(
                mask,
                jnp.roll(
                    a=dist_function[..., i],
                    shift=lattice.e[:, i],
                    axis=[k for k in range(lattice.D)],
                ),
                dist_function[..., i],
            )
        )
    return dist_function


@partial(jit, static_argnums=(0))
def _stream_coupled(
    lattices: List[Lattice], dist_functions: List[ArrayLike], masks: List[ArrayLike]
) -> Array:
    """Apply the streaming step to the distribution functions.

    Args:
        lattices (CoupledLattices): The coupled lattices on which the streaming is performed.
        dist_functions (List[ArrayLike]): The distribution functions to be streamed.
        masks (List[ArrayLike]): The masks where the streaming should be performed.

    Returns:
        Array: The streamed distribution functions.
    """

    for i in range(len(lattices)):
        dist_functions[i] = _stream(lattices[i], dist_functions[i], masks[i])

    return dist_functions


@partial(jit, static_argnums=(0))
def stream(
    lattice: Union[Lattice, CoupledLattices],
    dist_function: Union[ArrayLike, List[ArrayLike]],
    mask: Union[ArrayLike, List[ArrayLike]]
)-> Union[Array, List[Array]]:
    """Apply the streaming step to the distribution function.

    Args:
        lattice (Lattice): The lattice on which the streaming is performed.
        dist_function (ArrayLike): The distribution function to be streamed.
        mask (ArrayLike): The mask where the streaming should be performed.

    Returns:
        Array: The streamed distribution function.
    """
    if isinstance(lattice, CoupledLattices):
        dist_function = _stream_coupled(lattice.to_tuple(), dist_function, mask)
    else:
        dist_function = _stream(lattice, dist_function, mask)
    return dist_function
