from functools import partial
from typing import List, Union

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


@partial(jit, static_argnums=(0))
def collide(
    lattice: Union[Lattice, CoupledLattices],
    dist_function: Union[ArrayLike, List[ArrayLike]],
    omega: Union[float, List[float]],
    mask: Union[ArrayLike, List[ArrayLike]],
    **kwargs
) -> Union[Array, List[Array]]:
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
    if isinstance(lattice, Lattice):
        # Smagorinsky sgs model
        # m, u = lattice.get_macroscopics(dist_function)
        # dudx, dudy = jnp.gradient(u[..., 0])
        # dvdx, dvdy = jnp.gradient(u[..., 1])
        # nu_sgs = 0.2**2 * jnp.sqrt(2 * dudx**2 + (dudy + dvdx)**2 + dvdy**2)
        # omega = (omega / (1.0 + 3 * omega * nu_sgs))[..., jnp.newaxis]

        dist_function = _collide(dist_function, equilibrium, force, mask, omega)
    else:
        for i in range(len(omega)):
            dist_function[i] = _collide(
                dist_function[i], equilibrium[i], force[i], mask[i], omega[i]
            )
    return dist_function
