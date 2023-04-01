import abc
from collections import namedtuple
from multipledispatch import dispatch

from functools import partial
from typing import Tuple, Union, List
from jax import Array, jit
from jax import numpy as jnp
from jax.typing import ArrayLike

from rllbm.lbm import Stencil

__all__ = ["Lattice", "FluidLattice", "ThermalFluidLattice"]


class Lattice(abc.ABC):
    Macroscopics: namedtuple
    _stencil: Stencil

    def __init__(self, stencil: Stencil):
        self._stencil = stencil

    @property
    def name(self):
        return self._name
    
    @property
    def stencil(self):
        return self._stencil
    
    def __getattr__(self, name):
        try:
            return getattr(self.stencil, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    @partial(jit, static_argnums=(0, 2))
    def get_moment(self, dist_function: ArrayLike, order: int) -> Array:
        return self.stencil.get_moment(
            dist_function=dist_function,
            order=order,
        )

    @abc.abstractmethod
    def initialize(self, dist_function: ArrayLike, *args, **kwargs) -> Array:
        pass

    @abc.abstractmethod
    def force(self, *args, **kwargs) -> Union[Array, float]:
        pass

    @abc.abstractmethod
    def equilibrium(self, *args, **kwargs) -> Array:
        pass

    @abc.abstractmethod
    def collision_terms(self, dist_function: ArrayLike, *args, **kwards) -> Array:
        pass


class FluidLattice(Lattice):
    Macroscopics = namedtuple('FluidMacroscopics', ('rho', 'u'))

    def __init__(self, stencil: Stencil):
        super().__init__(stencil)

    def initialize(self, rho: ArrayLike, u: ArrayLike) -> Array:
        df_equilibrium = self.equilibrium(rho, u)
        return df_equilibrium

    @dispatch(Array, Array)
    @partial(jit, static_argnums=(0))
    def equilibrium(self, rho: ArrayLike, u: ArrayLike) -> Array:
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
    
    @dispatch(tuple)
    @partial(jit, static_argnums=(0))
    def equilibrium(self, fluid_state) -> Array:
        return self.equilibrium(fluid_state.rho, fluid_state.u)

    @partial(jit, static_argnums=(0))
    def force(self):
        return 0.0

    @partial(jit, static_argnums=(0))
    def collision_terms(self, dist_function: ArrayLike) -> Tuple[Array]:
        fluid_state = self.get_macroscopics(dist_function)
        df_equilibrium = self.equilibrium(fluid_state.rho, fluid_state.u)
        df_force = self.force()
        return df_equilibrium, df_force

    @partial(jit, static_argnums=(0))
    def get_macroscopics(self, dist_function: ArrayLike) -> Array:
        rho = self.get_moment(dist_function, order=0)[..., jnp.newaxis]
        u = self.get_moment(dist_function, order=1) / rho
        return self.Macroscopics(rho, u)


class CoupledLattices(abc.ABC):
    Macroscopics: namedtuple

    def __init__(self, coupled_lattice_dict: dict):
        self.lattices = coupled_lattice_dict
        
    def to_tuple(self):
        return tuple(self.lattices.values())

    def __getitem__(self, key) -> Lattice:
        return self.lattices[key]

    def __len__(self):
        return len(self.lattices)

    @abc.abstractmethod
    def initialize(
        self, dist_functions: List[ArrayLike], *args, **kwargs
    ) -> List[Array]:
        pass

    @abc.abstractmethod
    def force(self, *args, **kwargs) -> List[Union[float, Array]]:
        pass

    @abc.abstractmethod
    def equilibrium(self, *args, **kwargs) -> Tuple[Array]:
        pass

    @abc.abstractmethod
    def collision_terms(
        self, dist_function: List[ArrayLike], *args, **kwards
    ) -> List[Array]:
        pass


class ThermalFluidLattice(CoupledLattices):
    Macroscopics = namedtuple(f'ThermalFluidMacroscopics', ('rho', 'T', "u"))

    def __init__(self, fluid_stencil, thermal_stencil):
        coupled_lattice_dict = {
            "fluid": FluidLattice(fluid_stencil),
            "thermal": FluidLattice(thermal_stencil),
        }
        super().__init__(coupled_lattice_dict)

    @partial(jit, static_argnums=(0))
    def initialize(self, rho: ArrayLike, T: ArrayLike, u: ArrayLike) -> List[Array]:
        """Initialize the distribution function with the equilibrium distribution
        corresponding to the given density, temperature and velocity.

        Args:
            rho (ArrayLike): The prescribed density of the fluid.
            T (ArrayLike): The prescribed temperature of the fluid.
            u (ArrayLike): The prescribed velocity of the fluid.

        Returns:
            Array: The initialized distribution function.
        """
        return self.equilibrium(rho, T, u)

    @dispatch(Array, Array, Array)
    @partial(jit, static_argnums=(0))
    def equilibrium(self, rho: ArrayLike, T: ArrayLike, u: ArrayLike) -> List[Array]:
        """Computes the equilibrium distribution functions of the coupled lattixes

        Args:
            rho (ArrayLike): The density of the fluid.
            T (ArrayLike): The temperature of the fluid.
            u (ArrayLike): The velocity of the fluid.

        Returns:
            Tuple[Array]: The equilibrium distribution function for the fluid and
                thermal lattices.
        """

        fluid_eq = self["fluid"].equilibrium(rho, u)
        thermal_eq = self["thermal"].equilibrium(T, u)

        return [fluid_eq, thermal_eq]
    
    @dispatch(tuple)
    @partial(jit, static_argnums=(0))
    def equilibrium(self, fluid_state) -> Array:
        return self.equilibrium(fluid_state.rho, fluid_state.T, fluid_state.u)

    @partial(jit, static_argnums=(0))
    def force(
        self,
        rho: ArrayLike,
        T: ArrayLike,
        gravity: ArrayLike,
        thermal_expansion: float,
        timestep: float,
    ) -> Tuple[Array]:
        """Computes the force on the fluid due to gravity.

        Args:
            rho: The density of the fluid.
            T: The temperature of the fluid.
            gravity: The gravitational acceleration vector.
            thermal_expansion: The thermal expansion coefficient of the fluid.
            timestep: The timestep of the simulation.

        Returns:
            The force on the fluid due to gravity.
        """
        stencil = self["fluid"].stencil

        # Project the gravity vector onto the lattice directions
        e_dot_f = jnp.einsum("dQ, d->Q", stencil.e, gravity)[
            jnp.newaxis, jnp.newaxis, :
        ]

        scalar = timestep / stencil.cs**2 * thermal_expansion

        # Compute the force
        fluid_force = (
            scalar
            * rho
            * T
            * e_dot_f
            * stencil.w[jnp.newaxis, jnp.newaxis, :]
        )
        thermal_force = 0.0

        return fluid_force, thermal_force

    @partial(jit, static_argnums=(0))
    def collision_terms(
        self,
        dist_functions: List[ArrayLike],
        gravity: ArrayLike,
        thermal_expansion: float,
        timestep: float,
    ) -> Tuple:
        """Compute the collision terms for the coupled fluid and thermal lattices.

        Args:
            dist_functions (Tuple[ArrayLike]): The distribution functions of the
                fluid and thermal lattices.
            gravity (ArrayLike): The gravity.
            thermal_expansion (float): The thermal expansion coefficient.

        Returns:
            Tuple[Tuple[Array], Tuple[Array]]: The equilibrium and force terms for
                the fluid and thermal lattices.
        """
        # Get the moments of the distribution functions
        fluid_state = self.get_macroscopics(dist_functions)

        # Compute the equilibrium terms
        equilibrium = self.equilibrium(fluid_state.rho, fluid_state.T, fluid_state.u)

        # Compute the force terms
        force = self.force(
            rho=fluid_state.rho,
            T=fluid_state.T,
            gravity=gravity,
            thermal_expansion=thermal_expansion,
            timestep=timestep,
        )

        return equilibrium, force

    @partial(jit, static_argnums=(0))
    def get_macroscopics(self, dist_functions: List[ArrayLike]) -> Macroscopics:
        """Get the macroscopic quantities for the coupled NSE and ADE lattices.

        Args:
            dist_functions (List[ArrayLike]): The distribution functions of the
                fluid and thermal lattices.

        Returns:
            List[Array]: The macroscopic quantities for the NSE and ADE lattices.
        """
        rho = self["fluid"].get_moment(dist_functions[0], order=0)[..., jnp.newaxis]
        u = self["fluid"].get_moment(dist_functions[0], order=1) / rho
        T = self["thermal"].get_moment(dist_functions[1], order=0)[..., jnp.newaxis]

        return self.Macroscopics(rho, T, u)
