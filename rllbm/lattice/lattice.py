from functools import partial
import string

from jax import jit, Array
import jax.numpy as jnp
from jax.typing import ArrayLike



class Lattice:
    """_summary_
    """
    def __init__(
        self,
        coords: ArrayLike,
        weights: ArrayLike,
        name: str,
        dim: int,
        **kwargs,
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

    @partial(jit, static_argnums=2)
    def get_moment(self, dist_function, order):
        """_summary_

        Args:
            dist_function (_type_): _description_
            order (_type_): _description_

        Returns:
            _type_: _description_
        """
        lowercase = string.ascii_lowercase
        coord_litteral = "".join([f",{lowercase[i%26]}Q" for i in range(order)])
        output_litteral = "".join([f"{lowercase[i%26]}" for i in range(order)])
        
        einsum_litteral = "NMQ" + coord_litteral + "->NM" + output_litteral
        
        args = [self.coords] * order
        return jnp.einsum(einsum_litteral, dist_function, *args)
    
    def _tree_flatten(self):
        children = (self.coords, self.weights)  # arrays / dynamic values
        aux_data = {'name': self.name, 'dim': self.dim, 'size': self.size}  # static values
        return (children, aux_data)
    
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

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