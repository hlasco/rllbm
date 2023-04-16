from __future__ import annotations

from typing import TYPE_CHECKING, Union, Dict

import chex

from jax import numpy as jnp

from rllbm.lbm.lattice import Lattice, CoupledLattices

if TYPE_CHECKING:
    from rllbm.lbm.simulation import LBMState

__all__ = ["stream"]


def stream_df(lattice: Lattice, df: chex.Array, mask: chex.Array) -> chex.Array:
    """This function takes in a distribution function, and a mask, and streams the
    distribution function according to the mask. The mask allows to stream the distribution
    function only on the fluid nodes, and not on the boundary nodes.

    Args:
        lattice (Lattice): The lattice on which the streaming is performed.
        df (chex.Array,): The distribution function to be streamed.
        mask (chex.Array,): The mask where the streaming should be performed.

    Returns:
        Array: The streamed distribution function.
    """

    for i in range(lattice.Q):
        df = df.at[..., i].set(
            jnp.where(
                mask,
                jnp.roll(
                    a=df[..., i],
                    shift=lattice.e[:, i],
                    axis=[k for k in range(lattice.D)],
                ),
                df[..., i],
            )
        )
    return df


def stream(
    lattice: Union[Lattice, CoupledLattices], state_dict: Dict[str, LBMState]
) -> Dict[str, LBMState]:
    """ """
    for name in state_dict.keys():
        if isinstance(lattice, Lattice):
            l = lattice
        else:
            l = lattice.lattices[name]

        state_dict[name].df = stream_df(
            l, state_dict[name].df, state_dict[name].stream_mask
        )

    return state_dict
