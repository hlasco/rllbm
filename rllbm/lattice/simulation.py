from functools import partial
from typing import List, Union

from jax import Array, jit
from jax import numpy as jnp
from jax.typing import ArrayLike

from rllbm.lattice.lattice import Lattice, CoupledLattices
from rllbm.lattice.collisions import collide
from rllbm.lattice.boundary import BoundaryDict, apply_boundary_conditions
from rllbm.lattice.stream import stream

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

    def initialize(self, lattice, dfs, tracers=None) -> None:
        self.lattice = lattice
        self.dfs = dfs
        self.tracers = tracers
        self.time = 0

    def set_boundary_conditions(self, boundaries, bc_kwargs) -> None:
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
    def _step(self, dfs, bc_kwargs):
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
    def get_macroscopics(self, dfs):
        return self.lattice.get_macroscopics(dfs)

    # def _update_tracers(
    #    self,
    #    tracers,
    #    dist_function_NSE,
    # ):
    #    density = D2Q9.get_moment(dist_function_NSE, 0)
    #    velocity = D2Q9.get_moment(dist_function_NSE, 1) / density[..., jnp.newaxis]
    #    for tracer_id, tracer in enumerate(tracers):
    #        idx, idy = jnp.floor(tracer[0] / self.dx), jnp.floor(tracer[1] / self.dx)
    #        tracer += velocity[idx.astype(int)%self.nx, idy.astype(int)%self.ny, :] * self.dt
    #        tracer = tracer.at[0].set(tracer[0] % 1.0)
    #        tracer = tracer.at[1].set(tracer[1] % 1.0)
    #        tracers[tracer_id] = tracer
    #
    #    return tracers
