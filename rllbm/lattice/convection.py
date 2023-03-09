from rllbm.lattice import Lattice

from jax import Array
import jax.numpy as jnp
from jax.typing import ArrayLike
from typing import Tuple

def collide(
    dist_function_NSE: ArrayLike,
    lattice_NSE: Lattice,
    omega_NSE: jnp.float64,
    dist_function_TD: ArrayLike,
    lattice_TD: Lattice,
    omega_TD: jnp.float64,
    buoyancy: ArrayLike,
) -> Tuple[Array]:
    """_summary_

    Args:
        dist_function_NSE (ArrayLike): _description_
        lattice_NSE (Lattice): _description_
        omega_NSE (jnp.float64): _description_
        dist_function_TD (ArrayLike): _description_
        lattice_TD (Lattice): _description_
        omega_TD (jnp.float64): _description_
        buoyancy (ArrayLike): _description_

    Returns:
        Tuple[Array]: _description_
    """
    
    density = lattice_NSE.get_moment(dist_function_NSE, 0)[..., jnp.newaxis]
    velocity =  lattice_NSE.get_moment(dist_function_NSE, 1) / density
    velocity_norm2 = jnp.linalg.norm(velocity, axis=-1, ord=2,)[..., jnp.newaxis]**2
    
    temperature = lattice_TD.get_moment(dist_function_TD, 0)[..., jnp.newaxis]
    
    tmp = 3*jnp.einsum(
        "dQ, NMd->NMQ",
        lattice_NSE.coords,
        velocity,
    )
    
    eq_NSE =  (
        density * lattice_NSE.weights[jnp.newaxis, jnp.newaxis, :] * (
            1 + tmp + 1/2 * tmp**2 - 3/2 * velocity_norm2
        )
    )
    
    tmp = jnp.einsum(
        "dQ, d->Q",
        lattice_NSE.coords,
        buoyancy,
    )[jnp.newaxis, jnp.newaxis, :]
    
    force_NSE = (
        3 * density * lattice_NSE.weights[jnp.newaxis, jnp.newaxis, :] * (
            temperature * tmp
        )
    )
    
    tmp = 3*jnp.einsum(
        "dQ, NMd->NMQ",
        lattice_TD.coords,
        velocity,
    )
    
    eq_TD = (
        temperature * lattice_TD.weights[jnp.newaxis, jnp.newaxis, :] * (
            1 + tmp
        )
    )
    
    dist_function_NSE = dist_function_NSE - omega_NSE * (dist_function_NSE - eq_NSE) + force_NSE
    dist_function_TD = dist_function_TD - omega_TD * (dist_function_TD - eq_TD)
    
    return dist_function_NSE, dist_function_TD