from functools import partial
from collections.abc import Iterable

from multipledispatch import dispatch
from typing import Sequence, Union

import chex
import jax
from jax import numpy as jnp

from rllbm.lbm.lattice import Lattice, CoupledLattices

__all__ = ["collide"]


@jax.jit
def _collide(
    dist_function: chex.Array,
    equilibrium: chex.Array,
    force: chex.Array,
    mask: chex.Array,
    omega: Union[chex.Scalar, chex.Array],
) -> chex.Array:
    """Perform a BKG collision step.

    This function performs a BKGK collision of a single density function. It takes in
    the distribution function, the equilibrium distribution, the force,
    the mask, and the relaxation parameter, and returns the new distribution
    function.

    Args:
        dist_function (chex.Array): The distribution function to be collided.
        equilibrium (chex.Array): The equilibrium distribution.
        force (chex.Array): The force distribution.
        mask (chex.Array): The mask where the collision should be performed.
        omega (Union[chex.Scalar, chex.Array]): The relaxation parameter.

    Returns:
        Array: The new distribution function after collision
    """
    return jnp.where(
        mask[..., jnp.newaxis],
        dist_function - omega * (dist_function - equilibrium) + force,
        dist_function,
    )
    
@dispatch(Lattice, jax.Array, (jax.Array, float),  jax.Array)
@partial(jax.jit, static_argnums=(0))
def collide(
    lattice: Lattice,
    dist_function: chex.Array,
    omega: chex.Scalar,
    mask: chex.Array,
    **kwargs
) -> Union[chex.Array, Sequence[chex.Array]]:
    """Collide the distribution function on the lattice defined in the Lattice object.

    Args:
        lattice (Lattice): The Lattice object containing the lattice to collide.
        dist_function (chex.Array): The distribution function to collide.
        omega (float): The relaxation frequency.
        mask (chex.Array): The mask where the collision should be performed.
        **kwargs: Any additional keyword arguments to pass to the collision terms
            function.
    Returns:
        chex.Array: The new distribution function.
    """
    equilibrium, force = lattice.collision_terms(dist_function, **kwargs)
    dist_function = _collide(dist_function, equilibrium, force, mask, omega)


@dispatch(CoupledLattices, Iterable, Iterable, Iterable)
@partial(jax.jit, static_argnums=(0))
def collide(
    lattice: CoupledLattices,
    dist_function: Sequence[chex.Array],
    omega: Sequence[chex.Scalar],
    mask: Sequence[chex.Array],
    **kwargs
) -> Sequence[chex.Array]:
    """Collide the distribution function on the lattice defined in the Lattice object.

    Args:
        lattice (CoupledLattices): The Lattice object containing the lattice to collide.
        dist_function (Sequence[chex.Array]): The distribution function to collide.
        omega (Sequence[chex.Scalar]): The relaxation frequency.
        mask (Sequence[chex.Array]): The mask where the collision should be performed.
        **kwargs: Any additional keyword arguments to pass to the collision terms
            function.
    Returns:
        Sequence[chex.Array]: The new distribution function.
    """
    equilibrium, force = lattice.collision_terms(dist_function, **kwargs)
    for i in range(len(omega)):
        dist_function[i] = _collide(
            dist_function[i], equilibrium[i], force[i], mask[i], omega[i]
        )
    return dist_function
