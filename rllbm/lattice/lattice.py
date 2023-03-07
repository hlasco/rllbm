from dataclasses import dataclass

import jax.numpy as jnp

from jax import jit, Array
from jax.typing import ArrayLike

from functools import partial

class Lattice:
    """_summary_
    """
    def __init__(
        self,
        name: str,
        dim: int,
        coords: ArrayLike,
        weights: ArrayLike,
    ):
        """_summary_

        Args:
            name (str): _description_
            dim (int): _description_
            coords (ArrayLike): _description_
            weights (ArrayLike): _description_
        """
        self.name = name
        self.coords = coords
        self.weights = weights

        self.dim, self.size = self.coords.shape

        assert dim == self.dim, f"Expected coordinates with dimension {dim}, got: {self.dim}"

        self.opposite_indices = _get_opposite_indices(self.coords)

    def get_moment(self, dist_function, order):
        return _get_moment(dist_function, order)

@partial(jit, static_argnums=0)
def _get_moment(
    dist_function: ArrayLike,
    order: int,
) -> Array:
    """_summary_

    Args:
        dist_function (ArrayLike): _description_
        order (int): _description_

    Returns:
        Array: _description_
    """
    einsum_litteral = "NMQ" + order*",dQ" + "->NM" + order*"d"
    args = [coords] * order

    return jnp.einsum(einsum_litteral, dist_function, *args)

def _get_opposite_indices(
        coords: ArrayLike
    ) -> Array:
    """_summary_

    Args:
        coords (ArrayLike): _description_

    Returns:
        Array: _description_
    """
    dim, size = coords.shape

    identity = jnp.eye(dim)
    indices = jnp.empty(size)

    for i, direction in enumerate(coords.T):
        opposite_direction = - jnp.dot(identity, direction)
        index_match = jnp.prod(coords.T == opposite_direction[jnp.newaxis, :], axis=1)
        indices = indices.at[i].set(jnp.argwhere(index_match, size=1).squeeze())

    return indices