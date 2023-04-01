from functools import partial
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Tuple, List, Union

from jax import Array, jit
from jax import numpy as jnp
from jax.typing import ArrayLike

from rllbm.lbm.lattice import Lattice, CoupledLattices
from rllbm.lbm.collisions import collide
from rllbm.lbm.boundary import BoundaryDict, apply_boundary_conditions
from rllbm.lbm.stream import stream

__all__ = ["Domain", "Simulation"]

@dataclass
class Domain:
    shape: Tuple[int]
    bounds: Tuple[float]
    
    def __post_init__(self):
        self.dim = len(self.shape)
        
        if self.dim not in [1, 2, 3]:
            raise ValueError(
                f"The number of dimensions must be 1, 2 or 3, but got {len(self.shape)} instead."
            )
        
        if len(self.bounds) != 2*len(self.shape):
            raise ValueError(
                f"The number of bounds must be twice the number of dimensions, "
                f"but got {self.shape} and {self.bounds} instead."
            )
        for i in range(self.dim):
            low = self.dim*i
            high = low + 1
            if (self.bounds[low] > self.bounds[high]):
                raise ValueError(
                    f"The lower bound of dimension {i} must be smaller than the upper bound "
                    f"but got {self.bounds[low]} and {self.bounds[high]}."
                )
        self.width = [self.bounds[self.dim*i+1] - self.bounds[self.dim*i] for i in range(self.dim)]
        self.dx = min(self.width[i] / (self.shape[i]-1) for i in range(self.dim))
        
        X, Y = jnp.meshgrid(self.x, self.y, indexing="ij")
        
        self.left = (X == self.bounds[0])
        self.right = (X == self.bounds[1])
        self.bottom = (Y == self.bounds[2])
        self.top = (Y == self.bounds[3])
        
    @property
    def x(self):
        return jnp.linspace(self.bounds[0], self.bounds[1], self.shape[0])

    @property
    def y(self):
        return jnp.linspace(self.bounds[2], self.bounds[3], self.shape[1])
    
    @property
    def z(self):
        return jnp.linspace(self.bounds[4], self.bounds[5], self.shape[2])
        

class Simulation:
    def __init__(
        self,
        domain: Domain,
        omegas: Union[List[float], float],
        collision_kwargs: dict = {},
    ) -> None:
        self._domain = domain

        self.omegas = omegas
        self.collision_kwargs = collision_kwargs

        self.boundaries = None
        self.bc_kwargs = None
        self.lattice = None
        self.dfs = None
        self.collision_mask = None
        self.stream_mask = None
        
    @property
    def domain(self):
        return self._domain
    
    def __getattr__(self, name):
        try:
            return getattr(self._domain, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def initialize(
        self, lattice: Union[Lattice, CoupledLattices], dfs: Union[ArrayLike, List[ArrayLike]],
    ) -> None:
        self.lattice = lattice
        self.dfs = dfs

    def set_boundary_conditions(
        self, boundaries: Union[BoundaryDict, List[BoundaryDict]], bc_kwargs: Union[dict, List[dict]]
    ) -> None:
        self.boundaries = boundaries
        self.bc_kwargs = bc_kwargs

        if isinstance(boundaries, Iterable):
            self.collision_mask = [bdy.collision_mask for bdy in self.boundaries]
            self.stream_mask = [bdy.stream_mask for bdy in self.boundaries]
        else:
            self.collision_mask = self.boundaries.collision_mask
            self.stream_mask = self.boundaries.stream_mask

    def step(self):
        self.dfs = self._step(self.dfs, self.bc_kwargs)

    @partial(jit, static_argnums=(0))
    def _step(
        self, dfs: Union[ArrayLike, List[ArrayLike]], bc_kwargs: Union[dict, List[dict]]
    ) -> Union[Array, List[Array]]:

        dfs = collide(
            self.lattice,
            dfs,
            self.omegas,
            self.collision_mask,
            **self.collision_kwargs,
        )

        dfs = stream(self.lattice, dfs, self.stream_mask)

        dfs = apply_boundary_conditions(self.lattice, self.boundaries, dfs, bc_kwargs)

        return dfs

    @partial(jit, static_argnums=(0))
    def get_macroscopics(self, dfs: ArrayLike) -> List[Array]:
        return self.lattice.get_macroscopics(dfs)
