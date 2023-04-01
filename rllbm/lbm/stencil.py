import string
from functools import partial

import chex
import jax

from jax import numpy as jnp

__all__ = ["Stencil", "D1Q3", "D2Q5", "D2Q9"]


class Stencil:
    e: chex.Array = jnp.array([])
    w: chex.Array = jnp.array([])
    opposite: chex.Array = jnp.array([])
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
    @partial(jax.jit, static_argnums=(0, 2))
    def get_moment(cls, dist_function: chex.Array, order: int) -> chex.Array:
        """Returns the moment of the distribution function.
        """
        dim = len(dist_function.shape) - 1
        
        # Create the einsum litteral
        lowercase = string.ascii_lowercase
        uppercase = string.ascii_uppercase
        
        e_litteral = "".join([f",{lowercase[i%26]}Q" for i in range(order)])
        d_litteral = "".join([f"{uppercase[i%26]}" for i in range(dim)])
        
        output_litteral = "".join([f"{lowercase[i%26]}" for i in range(order)])

        # The einsum litteral is of the form: "AB,Qa,Qb->ABab"
        einsum_litteral = d_litteral + "Q" + e_litteral + "->" + d_litteral + output_litteral

        # Get the distribution function moments
        stacked_e = [cls.e] * order
        return jnp.einsum(einsum_litteral, dist_function, *stacked_e)


class D1Q3(Stencil):
    r"""
    Stencil: D2Q5
        2 - 0 - 1
    """
    e = jnp.array([[0, 1, -1]])
    w = jnp.array([2.0 / 3, 1.0 / 6, 1.0 / 6])
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
    e = jnp.array([[0, 1, 0, -1, 0], [0, 0, 1, 0, -1]])
    w = jnp.array([1.0 / 3, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6])
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
    e = jnp.array([[0, 1, 0, -1, 0, 1, -1, -1, 1], [0, 0, 1, 0, -1, 1, 1, -1, -1]])
    w = jnp.array(
        [
            4.0 / 9,
            1.0 / 9,
            1.0 / 9,
            1.0 / 9,
            1.0 / 9,
            1.0 / 36,
            1.0 / 36,
            1.0 / 36,
            1.0 / 36,
        ]
    )
    opposite = jnp.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
    cs = 1 / jnp.sqrt(3)
