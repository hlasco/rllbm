from __future__ import annotations

import abc
from collections import namedtuple
from collections.abc import Iterable
from multipledispatch import dispatch

from functools import partial
from typing import TYPE_CHECKING, Sequence, Union, Dict

import chex
import jax
from jax import numpy as jnp

from rllbm.lbm import Stencil

if TYPE_CHECKING:
    from rllbm.lbm.simulation import LBMState

__all__ = ["Lattice", "FluidLattice", "ThermalFluidLattice"]


class Lattice(abc.ABC):
    Macroscopics: namedtuple
    name: str
    _stencil: Stencil

    def __init__(self, stencil: Stencil, name: str):
        self._stencil = stencil
        self.name = name

    @property
    def stencil(self):
        return self._stencil

    def __getattr__(self, name):
        try:
            return getattr(self.stencil, name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

    @partial(jax.jit, static_argnums=(0, 2))
    def get_moment(self, dist_function: chex.Array, order: int) -> chex.Array:
        return self.stencil.get_moment(
            dist_function=dist_function,
            order=order,
        )

    @abc.abstractmethod
    def initialize(self, *args, **kwargs) -> chex.Array:
        pass

    @abc.abstractmethod
    def force(self, *args, **kwargs) -> Union[chex.Array, chex.Scalar]:
        pass

    @abc.abstractmethod
    def equilibrium(self, *args, **kwargs) -> chex.Array:
        pass

    @abc.abstractmethod
    def collision_terms(self, state_dict) -> chex.Array:
        pass


class FluidLattice(Lattice):
    Macroscopics = namedtuple("FluidMacroscopics", ("rho", "u"))

    def __init__(self, stencil: Stencil, name: str = "FluidLattice"):
        super().__init__(stencil, name)

    def initialize(self, rho: chex.Array, u: chex.Array) -> chex.Array:
        return self.equilibrium(rho, u)

    @dispatch(jax.Array, jax.Array)
    @partial(jax.jit, static_argnums=(0))
    def equilibrium(self, rho: chex.Array, u: chex.Array) -> chex.Array:
        u_norm2 = (
            jnp.linalg.norm(
                u,
                axis=-1,
                ord=2,
            )[..., jnp.newaxis]
            ** 2
        )

        if u.ndim == 2:
            # Flattened array of velocities (Nxd)
            w = self.w[jnp.newaxis, :]
            e_dot_u = jnp.einsum("dQ, Xd->XQ", self.e, u)
        elif u.ndim == 3:
            # 2D array of 2D velocities (XxYx2)
            w = self.w[jnp.newaxis, jnp.newaxis, :]
            e_dot_u = jnp.einsum("dQ, XYd->XYQ", self.e, u)
        elif u.ndim == 4:
            # 3D array of 3D velocities (XxYxZx3)
            w = self.w[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
            e_dot_u = jnp.einsum("dQ, XYZd->XYZQ", self.e, u)
        else:
            raise ValueError("velocity must be 2D, 3D or 4D array")

        cs = self.cs

        df_eq = (
            rho
            * w
            * (
                1
                + e_dot_u / cs**2
                + 0.5 * e_dot_u**2 / cs**4
                - 0.5 * u_norm2 / cs**2
            )
        )

        return df_eq

    @dispatch(Iterable)
    @partial(jax.jit, static_argnums=(0))
    def equilibrium(self, fluid_state: Sequence[chex.Array]) -> chex.Array:
        return self.equilibrium(fluid_state.rho, fluid_state.u)

    @partial(jax.jit, static_argnums=(0))
    def force(self):
        return 0.0

    @partial(jax.jit, static_argnums=(0))
    def collision_terms(self, fluid_state) -> Sequence[chex.Array]:
        df_equilibrium = self.equilibrium(fluid_state.rho, fluid_state.u)
        df_force = self.force()
        return df_equilibrium, df_force

    @dispatch(jax.Array)
    @partial(jax.jit, static_argnums=(0))
    def get_macroscopics(self, df: chex.Array) -> Sequence[chex.Array]:
        rho = self.get_moment(df, order=0)[..., jnp.newaxis]
        u = self.get_moment(df, order=1) / rho
        return self.Macroscopics(rho, u)

    @dispatch(dict)
    @partial(jax.jit, static_argnums=(0))
    def get_macroscopics(self, state_dict: Dict[str, LBMState]) -> Sequence[chex.Array]:
        df = state_dict[self.name].df
        return self.get_macroscopics(df)


class CoupledLattices(abc.ABC):
    Macroscopics: namedtuple

    def __init__(self, coupled_lattice_dict: dict):
        self.lattices = coupled_lattice_dict

    def to_tuple(self):
        return tuple(self.lattices.values())

    def keys(self):
        return self.lattices.keys()

    def __iter__(self):
        return iter(self.lattices.items())

    def __getitem__(self, key) -> Lattice:
        return self.lattices[key]

    def __len__(self):
        return len(self.lattices)

    @abc.abstractmethod
    def initialize(self, *args, **kwargs) -> Sequence[chex.Array]:
        pass

    @abc.abstractmethod
    def force(self, *args, **kwargs) -> Sequence[Union[chex.Scalar, chex.Array]]:
        pass

    @abc.abstractmethod
    def equilibrium(self, *args, **kwargs) -> Sequence[chex.Array]:
        pass

    @abc.abstractmethod
    def collision_terms(
        self, dist_function: Sequence[chex.Array], *args, **kwards
    ) -> Sequence[chex.Array]:
        pass


class ThermalFluidLattice(CoupledLattices):
    Macroscopics = namedtuple(f"ThermalFluidMacroscopics", ("rho", "T", "u"))

    def __init__(
        self, fluid_stencil, thermal_stencil, buoyancy,
    ):
        coupled_lattice_dict = {
            "FluidLattice": FluidLattice(fluid_stencil, "Fluid"),
            "ThermalLattice": FluidLattice(thermal_stencil, "Thermal"),
        }
        self.buoyancy = buoyancy
        super().__init__(coupled_lattice_dict)

    @partial(jax.jit, static_argnums=(0))
    def initialize(
        self, rho: chex.Array, T: chex.Array, u: chex.Array
    ) -> Sequence[chex.Array]:
        return self.equilibrium(rho, T, u)

    @dispatch(jax.Array, jax.Array, jax.Array)
    @partial(jax.jit, static_argnums=(0))
    def equilibrium(
        self, rho: chex.Array, T: chex.Array, u: chex.Array
    ) -> Sequence[chex.Array]:
        """Computes the equilibrium distribution functions of the coupled lattixes

        Args:
            rho (chex.Array): The density of the fluid.
            T (chex.Array): The temperature of the fluid.
            u (chex.Array): The velocity of the fluid.

        Returns:
            Sequence[chex.Array]: The equilibrium distribution function for the fluid and
                thermal lattices.
        """
        ret = {
            "FluidLattice": self["FluidLattice"].equilibrium(rho, u),
            "ThermalLattice": self["ThermalLattice"].equilibrium(T, u),
        }

        return ret

    @dispatch(Iterable)
    @partial(jax.jit, static_argnums=(0))
    def equilibrium(self, fluid_state) -> Sequence[chex.Array]:
        return self.equilibrium(fluid_state.rho, fluid_state.T, fluid_state.u)

    @partial(jax.jit, static_argnums=(0))
    def force(
        self,
        rho: chex.Array,
        T: chex.Array,
    ) -> Sequence[Union[chex.Scalar, chex.Array]]:
        """Computes the force on the fluid due to gravity.

        Args:
            rho: The density of the fluid.
            T: The temperature of the fluid.

        Returns:
            The force on the fluid due to gravity.
        """
        stencil = self["FluidLattice"].stencil
        cs = stencil.cs

        # Project the gravity vector onto the lattice directions
        e_dot_b = jnp.einsum("dQ, d->Q", stencil.e, self.buoyancy)
        e_dot_b = e_dot_b[jnp.newaxis, jnp.newaxis, :]

        # Compute the force
        fluid_force = (
            rho * T * e_dot_b * stencil.w[jnp.newaxis, jnp.newaxis, :] / cs ** 2
        )
        thermal_force = 0.0

        ret = {
            "FluidLattice": fluid_force,
            "ThermalLattice": thermal_force,
        }

        return ret

    @partial(jax.jit, static_argnums=(0))
    def collision_terms(self, fluid_state) -> Sequence:
        """Compute the collision terms for the coupled fluid and thermal lattices.

        Args:
            dist_functions (Sequence[chex.Array]): The distribution functions of the
                fluid and thermal lattices.

        Returns:
            Sequence: The equilibrium and force terms for
                the fluid and thermal lattices.
        """

        # Compute the equilibrium terms
        equilibrium = self.equilibrium(fluid_state.rho, fluid_state.T, fluid_state.u)

        # Compute the force terms
        force = self.force(rho=fluid_state.rho, T=fluid_state.T)

        return equilibrium, force

    @partial(jax.jit, static_argnums=(0))
    def get_macroscopics(self, state_dict: Dict[str, LBMState]) -> Sequence[chex.Array]:
        """Get the macroscopic quantities for the coupled NSE and ADE lattices.

        Args:
            dist_functions (Sequence[chex.Array]): The distribution functions of the
                fluid and thermal lattices.

        Returns:
            Sequence[chex.Array]: The macroscopic quantities for the NSE and ADE lattices.
        """
        df_fluid = state_dict["FluidLattice"].df
        df_thermal = state_dict["ThermalLattice"].df

        rho = self["FluidLattice"].get_moment(df_fluid, order=0)[..., jnp.newaxis]
        u = self["FluidLattice"].get_moment(df_fluid, order=1) / rho
        T = self["ThermalLattice"].get_moment(df_thermal, order=0)[..., jnp.newaxis]

        return self.Macroscopics(rho, T, u)
