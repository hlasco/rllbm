from typing import Sequence
from collections.abc import Iterable
from functools import partial
from multipledispatch import dispatch

import chex
import jax

from jax import numpy as jnp

from rllbm.lbm.lattice import Lattice, CoupledLattices

__all__ = ["stream"]


@dispatch(Lattice, jax.Array, jax.Array)
@partial(jax.jit, static_argnums=(0))
def stream(lattice: Lattice, dist_function: chex.Array, mask: chex.Array,) -> chex.Array:
    """This function takes in a distribution function, and a mask, and streams the
    distribution function according to the mask. The mask allows to stream the distribution
    function only on the fluid nodes, and not on the boundary nodes.

    Args:
        lattice (Lattice): The lattice on which the streaming is performed.
        dist_function (chex.Array,): The distribution function to be streamed.
        mask (chex.Array,): The mask where the streaming should be performed.

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

@dispatch(CoupledLattices, Iterable, Iterable)
@partial(jax.jit, static_argnums=(0))
def stream(
    lattices: CoupledLattices, dist_functions: Sequence[chex.Array], masks: Sequence[chex.Array]
) -> Sequence[chex.Array]:
    """Apply the streaming step to the distribution functions.

    Args:
        lattices (CoupledLattices): The coupled lattices on which the streaming is performed.
        dist_functions (Sequence[ArrayLike]): The distribution functions to be streamed.
        masks (Sequence[ArrayLike]): The masks where the streaming should be performed.

    Returns:
        Array: The streamed distribution functions.
    """
    lattice_tuple = lattices.to_tuple()
    for i in range(len(lattice_tuple)):
        dist_functions[i] = stream(lattice_tuple[i], dist_functions[i], masks[i])

    return dist_functions
