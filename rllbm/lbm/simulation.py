from functools import partial
from typing import Union, Sequence, Dict
from dataclasses import dataclass

import chex
import jax

from jax import jit
from jax import numpy as jnp

from rllbm.lbm.lattice import Lattice, CoupledLattices
from rllbm.lbm.collisions import collide
from rllbm.lbm.boundary import Boundary, BoundaryDict, apply_boundary_conditions
from rllbm.lbm.stream import stream

__all__ = ["Domain", "Simulation", "LBMState"]


@chex.dataclass
class LBMState:
    df: chex.Array
    bc: BoundaryDict
    omega: chex.Scalar

    @property
    def collision_mask(self):
        return self.bc.collision_mask

    @property
    def stream_mask(self):
        return self.bc.stream_mask


@dataclass
class Domain:
    shape: Sequence[int]
    bounds: Sequence[float]

    def __post_init__(self):
        self.dim = len(self.shape)

        if self.dim not in [1, 2, 3]:
            raise ValueError(
                f"The number of dimensions must be 1, 2 or 3, but got {len(self.shape)} instead."
            )

        if len(self.bounds) != 2 * len(self.shape):
            raise ValueError(
                f"The number of bounds must be twice the number of dimensions, "
                f"but got {self.shape} and {self.bounds} instead."
            )
        for i in range(self.dim):
            low = self.dim * i
            high = low + 1
            if self.bounds[low] > self.bounds[high]:
                raise ValueError(
                    f"The lower bound of dimension {i} must be smaller than the upper bound "
                    f"but got {self.bounds[low]} and {self.bounds[high]}."
                )
        self.width = [
            self.bounds[self.dim * i + 1] - self.bounds[self.dim * i]
            for i in range(self.dim)
        ]
        self.dx = min(self.width[i] / (self.shape[i] - 1) for i in range(self.dim))

        X, Y = jnp.meshgrid(self.x, self.y, indexing="ij")

        self.left = X == self.bounds[0]
        self.right = X == self.bounds[1]
        self.bottom = Y == self.bounds[2]
        self.top = Y == self.bounds[3]

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
        lattice: Union[Lattice, CoupledLattices],
        omegas: Union[chex.Scalar, Dict],
    ) -> None:
        self.domain = domain

        self.fluid_state = None
        self.state_dict = {}
        self.lattice = lattice

        if isinstance(lattice, Lattice):
            self.state_dict[lattice.name] = LBMState(
                df=None,
                omega=omegas,
                bc=BoundaryDict(),
            )
        elif isinstance(lattice, CoupledLattices):
            for name in lattice.keys():
                self.state_dict[name] = LBMState(
                    df=None,
                    omega=omegas[name],
                    bc=BoundaryDict(),
                )
        else:
            raise TypeError(
                f"The lattice must be an instance of Lattice or CoupledLattices, "
                f"but got {type(lattice)} instead."
            )

    def __getattr__(self, name):
        try:
            return getattr(self.domain, name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute {name}"
            )

    def set_initial_conditions(self, **kwargs) -> None:
        df = self.lattice.initialize(**kwargs)
        if isinstance(self.lattice, CoupledLattices):
            for lattice_name in self.lattice.keys():
                self.state_dict[lattice_name].df = df[lattice_name]
        else:
            self.state_dict[self.lattice.name].df = df

        self.fluid_state = self.lattice.get_macroscopics(self.state_dict)

    def set_boundary_conditions(
        self,
        bc: Union[Boundary, Sequence[Boundary]],
        lattice_name: str = None,
    ) -> None:
        if isinstance(self.lattice, CoupledLattices):
            if lattice_name is None:
                raise TypeError(
                    "The simulation lattice is a CoupledLattices, you must provide the lattice_name "
                    "to set boundary conditions."
                )
        else:
            lattice_name = self.lattice.name
        self.state_dict[lattice_name].bc.add(bc)

    def update_boundary_condition(
        self, bc_name: str, params: Dict, lattice_name: str = None
    ) -> None:
        if isinstance(self.lattice, CoupledLattices):
            if lattice_name is None:
                raise TypeError(
                    "The simulation lattice is a CoupledLattices, you must provide the lattice_name "
                    "to update boundary conditions."
                )
        else:
            lattice_name = self.lattice.name
        self.state_dict[lattice_name].bc.set_params(bc_name, params)

    def step(self):
        self.state_dict, self.fluid_state = self._step(self.state_dict)

    @partial(jit, static_argnums=(0))
    def _step(self, state_dict: Dict[str, LBMState]) -> Dict[str, LBMState]:
        state_dict = apply_boundary_conditions(self.lattice, state_dict)
        
        fluid_state = self.lattice.get_macroscopics(state_dict)
        state_dict = collide(self.lattice, state_dict, fluid_state)
        state_dict = stream(self.lattice, state_dict)
        
        return state_dict, fluid_state
