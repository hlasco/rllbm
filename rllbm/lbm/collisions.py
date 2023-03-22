from functools import partial
from typing import Tuple, overload, List

from jax import Array, jit
from jax import numpy as jnp
from jax.typing import ArrayLike

from rllbm.lbm.lattice import Lattice, CoupledLattices

__all__ = ["collide"]


@jit
def _collide(
    dist_function: ArrayLike,
    equilibrium: ArrayLike,
    force: ArrayLike,
    mask: ArrayLike,
    omega: float,
) -> Array:
    """Perform a BKG collision step.

    This function performs a BKGK collision of a single density function. It takes in
    the distribution function, the equilibrium distribution, the force,
    the mask, and the relaxation parameter, and returns the new distribution
    function.

    Args:
        dist_function (ArrayLike): The distribution function to be collided.
        equilibrium (ArrayLike): The equilibrium distribution.
        force (ArrayLike): The force distribution.
        mask (ArrayLike): The mask where the collision should be performed.
        omega (float): The relaxation parameter.

    Returns:
        Array: The new distribution function after collision
    """
    return jnp.where(
        mask[..., jnp.newaxis],
        dist_function - omega * (dist_function - equilibrium) + force,
        dist_function,
    )


@overload
@partial(jit, static_argnums=(0))
def collide(
    lattice: Lattice, dist_function: ArrayLike, omega: float, mask: ArrayLike, **kwargs
) -> Array:
    """Collide the distribution function on the lattice defined in the Lattice object.

    Args:
        lattice (Lattice): The Lattice object containing the lattice to collide.
        dist_function (ArrayLike): The distribution function to collide.
        omega (float): The relaxation frequency.
        mask (ArrayLike): The mask where the collision should be performed.
        **kwargs: Any additional keyword arguments to pass to the collision terms
            function.
    Returns:
        Array: The new distribution function.
    """
    equilibrium, force = lattice.collision_terms(dist_function, **kwargs)
    dist_function = _collide(dist_function, equilibrium, force, mask, omega)
    return dist_function


@partial(jit, static_argnums=(0))
def collide(
    lattices: CoupledLattices,
    dist_function: List[ArrayLike],
    omegas: Tuple[float],
    masks: Tuple[jnp.array],
    **kwargs
) -> List[ArrayLike]:
    """Collide the distribution functions on the lattices defined in the CoupledLattices
    object.

    This function performs a BKGK collision of a list of distribution functions. It
    takes in the distribution functions, the equilibrium distributions, the forces, the
    masks, and the relaxation parameters, and returns the new distribution functions.

    Args:
        lattices (CoupledLattices): The CoupledLattices object containing the lattices
            to collide.
        dist_function (List[ArrayLike]): The list of the distributions to
            collide.
        omegas (Tuple[float]): A tuple containing the relaxation frequencies for each
            of the lattices in the CoupledLattices object.
        masks (Tuple[jnp.array]): A tuple containing the masks for each of the lattices
            in the CoupledLattices object.
        **kwargs: Any additional keyword arguments to pass to the collision terms
            function.

    Returns:
        List[ArrayLike]: A list containing the new distributions.
    """
    equilibiurm, force = lattices.collision_terms(dist_function, **kwargs)

    for i in range(len(omegas)):
        dist_function[i] = _collide(
            dist_function[i], equilibiurm[i], force[i], masks[i], omegas[i]
        )
    return dist_function
