from __future__ import annotations
from functools import partial

from typing import TYPE_CHECKING, Dict, Union

import jax
from jax import numpy as jnp

from rllbm.lbm.lattice import Lattice, CoupledLattices

if TYPE_CHECKING:
    from rllbm.lbm.simulation import LBMState

__all__ = ["collide"]


@partial(jax.jit, static_argnums=(0,))
def collide(
    lattice: Union[Lattice, CoupledLattices], state_dict: Dict[str, LBMState]
) -> Dict[str, LBMState]:
    """Perform a BKG collision step."""
    eqs, forces = lattice.collision_terms(state_dict)

    # Smagorinsky sgs model
    # fluid_state = lattice.get_macroscopics(state_dict)
    # dudx, dudy = jnp.gradient(fluid_state.u[..., 0])
    # dvdx, dvdy = jnp.gradient(fluid_state.u[..., 1])
    # nu_sgs = 0.2**2 * jnp.sqrt(2 * dudx**2 + (dudy + dvdx)**2 + dvdy**2)
    # omega = (omega / (1.0 + 3 * omega * nu_sgs))[..., jnp.newaxis]

    for name in state_dict.keys():
        if isinstance(lattice, Lattice):
            eq, force = eqs, forces
        else:
            eq, force = eqs[name], forces[name]
        state_dict[name].df = jnp.where(
            state_dict[name].collision_mask[..., jnp.newaxis],
            state_dict[name].df
            - state_dict[name].omega * (state_dict[name].df - eq)
            + force,
            state_dict[name].df,
        )

    return state_dict
