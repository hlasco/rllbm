from functools import partial
from typing import List, Union

from jax import Array, jit
from jax import numpy as jnp
from jax.typing import ArrayLike

from rllbm.lbm.lattice import Lattice, CoupledLattices
from rllbm.lbm.collisions import collide
from rllbm.lbm.boundary import BoundaryDict, apply_boundary_conditions
from rllbm.lbm.stream import stream

__all__ = ["Simulation"]


class Simulation:
    def __init__(
        self,
        nx: int,
        ny: int,
        dt: float,
        omegas: Union[List[float], float],
        collision_kwargs: dict,
    ) -> None:
        self.nx = nx
        self.ny = ny
        self.dx = 1.0 / (max(nx, ny) - 1)
        self.x = jnp.arange(nx)
        self.y = jnp.arange(ny)
        self.dt = dt
        
        self.omegas = omegas
        self.collision_kwargs = collision_kwargs

        self.time = 0

        self.boundaries = None
        self.bc_kwargs = None
        self.lattice = None
        self.dfs = None
        self.collision_mask = None
        self.stream_mask = None

    def initialize(self, lattice: Union[Lattice, CoupledLattices], dfs: ArrayLike) -> None:
        self.lattice = lattice
        self.dfs = dfs
        self.time = 0

    def set_boundary_conditions(self, boundaries: BoundaryDict, bc_kwargs: dict) -> None:
        self.boundaries = boundaries
        self.bc_kwargs = bc_kwargs

        if isinstance(boundaries, tuple):
            self.collision_mask = [bdy.collision_mask for bdy in self.boundaries]
            self.stream_mask = [bdy.stream_mask for bdy in self.boundaries]
        else:
            self.collision_mask = self.boundaries.collision_mask
            self.stream_mask = self.boundaries.stream_mask

    def step(self):
        self.dfs = self._step(self.dfs, self.bc_kwargs)
        self.time += self.dt

    @partial(jit, static_argnums=(0))
    def _step(self, dfs: ArrayLike, bc_kwargs: dict) -> Array:
        dfs = collide(
            self.lattice,
            dfs,
            self.omegas,
            self.collision_mask,
            **self.collision_kwargs,
        )

        dfs = stream(
            self.lattice,
            dfs,
            self.stream_mask,
        )

        dfs = apply_boundary_conditions(self.lattice, self.boundaries, dfs, **bc_kwargs)

        return dfs

    @partial(jit, static_argnums=(0))
    def get_macroscopics(self, dfs: ArrayLike) -> List[Array]:
        return self.lattice.get_macroscopics(dfs)

