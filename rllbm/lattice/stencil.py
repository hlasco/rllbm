import string
from functools import partial

from jax import numpy as jnp
from jax.typing import ArrayLike
from jax import Array, jit

__all__ = ["Stencil", "D1Q3", "D2Q5", "D2Q9"]

class Stencil:
    e: ArrayLike = jnp.array([])
    w: ArrayLike = jnp.array([])
    opposite: ArrayLike = jnp.array([])
    cs: float = 0.0

    @classmethod
    @property
    def D(cls) -> int:
        return cls.e.shape[0]

    @classmethod
    @property
    def Q(cls) -> int:
        return cls.e.shape[1]

    @classmethod
    @partial(jit, static_argnums=(0, 2))
    def get_moment(
        cls,
        dist_function: ArrayLike,
        order: int
    ) -> Array:
        """Returns the moment of the distribution function.

        Args:
            dist_function (ArrayLike): _description_
            order (int): _description_

        Returns:
            Array: _description_
        """
        # Create the einsum litteral
        lowercase = string.ascii_lowercase
        e_litteral = "".join([f",{lowercase[i%26]}Q" for i in range(order)])
        output_litteral = "".join([f"{lowercase[i%26]}" for i in range(order)])
        
        einsum_litteral = "NMQ" + e_litteral + "->NM" + output_litteral
        
        # Get the distribution function moments
        args = [cls.e] * order
        return jnp.einsum(einsum_litteral, dist_function, *args)
    
class D1Q3(Stencil):
    r"""
    Stencil: D2Q5
        2 - 0 - 1
    """
    e = jnp.array([[0, 1, -1]])
    w = jnp.array([2./3, 1./6, 1./6])
    opposite = [0, 2, 1]
    cs = 1 / jnp.sqrt(3)
    
    
class D2Q5(Stencil):
    r"""
    Stencil: D2Q5
            2
            |
        3 - 0 - 1
            |
            4 
    """
    e = jnp.array([[0, 1, 0,-1, 0],
                   [0, 0, 1, 0,-1]])
    w = jnp.array([1./3, 1./6, 1./6, 1./6, 1./6])
    opposite = jnp.array([0, 3, 4, 1, 2])
    cs = 1 / jnp.sqrt(3)
    
class D2Q9(Stencil):
    r"""
    Stencil: D2Q9
        6   2   5
          \ | /
        3 - 0 - 1
          / | \
        7   4   8 
    """
    e = jnp.array([[0, 1, 0,-1, 0, 1,-1,-1, 1],
                  [0, 0, 1, 0,-1, 1, 1,-1,-1]])
    w = jnp.array([4./9, 1./9, 1./9, 1./9, 1./9, 1./36, 1./36, 1./36, 1./36])
    opposite = jnp.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
    cs = 1 / jnp.sqrt(3)
