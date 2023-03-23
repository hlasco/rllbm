import abc
from functools import partial
from typing import Tuple, Union, List

from jax import Array, jit
from jax import numpy as jnp
from jax.typing import ArrayLike

from rllbm.lbm import Stencil

__all__ = ["Lattice", "FluidLattice", "ConvectionLattice"]


class Lattice(abc.ABC, Stencil):
    _stencil: Stencil
    _shape: Tuple[int]
    _name: str = "BaseLattice"
    _macroscopics: List[str] = []

    def __init__(self, stencil: Stencil, shape: Tuple[int]):
        self._stencil = stencil
        self._shape = shape

    @property
    def name(self):
        return self._name

    @property
    def stencil(self):
        return self._stencil

    @property
    def e(self):
        return self._stencil.e

    @property
    def cs(self):
        return self._stencil.cs

    @property
    def w(self):
        return self._stencil.w

    @property
    def D(self):
        return self._stencil.D

    @property
    def Q(self):
        return self._stencil.Q

    @property
    def shape(self):
        return self._shape

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
    _name = "Fluid"

    def __init__(
        self,
        stencil: Stencil,
        shape: Tuple[int],
    ):
        super().__init__(stencil, shape)

    def initialize(self, m: ArrayLike, u: ArrayLike) -> Array:
        df_equilibrium = self.equilibrium(m, u)
        return df_equilibrium

    @partial(jit, static_argnums=(0))
    def equilibrium(self, m: ArrayLike, u: ArrayLike) -> Array:
        u_shape = u.shape
        u_norm2 = (
            jnp.linalg.norm(
                u,
                axis=-1,
                ord=2,
            )[..., jnp.newaxis]
            ** 2
        )

        if len(u_shape) == 2:
            w = self.w[jnp.newaxis, :]
            e_dot_u = jnp.einsum("dQ, ND->NQ", self.e, u)
        elif len(u_shape) == 3:
            w = self.w[jnp.newaxis, jnp.newaxis, :]
            e_dot_u = jnp.einsum("dQ, NMd->NMQ", self.e, u)
        else:
            raise ValueError("velocity must be 2D or 3D array")

        cs = self.cs

        df_eq = (
            m
            * w
            * (
                1
                + e_dot_u / cs**2
                + 0.5 * e_dot_u**2 / cs**4
                - 0.5 * u_norm2 / cs**2
            )
        )

        return df_eq

    @partial(jit, static_argnums=(0))
    def force(self):
        return 0.0

    @partial(jit, static_argnums=(0))
    def collision_terms(self, dist_function: ArrayLike) -> Tuple[Array]:
        m = self.get_moment(dist_function, order=0)[..., jnp.newaxis]
        u = self.get_moment(dist_function, order=1) / m

        df_equilibrium = self.equilibrium(m, u)
        df_force = self.force()

        return df_equilibrium, df_force

    @partial(jit, static_argnums=(0))
    def get_macroscopics(self, dist_function: ArrayLike) -> Array:
        m = self.get_moment(dist_function, order=0)[..., jnp.newaxis]
        u = self.get_moment(dist_function, order=1) / m
        return m, u


class CoupledLattices(abc.ABC):
    _name: str = "BaseCoupledLattices"

    def __init__(self, lattices: List[Lattice]):
        self.lattices = [l for l in lattices]

    def __getitem__(self, idx: int) -> Lattice:
        return self.lattices[idx]

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


class ConvectionLattice(CoupledLattices):
    _name = "ThermalFluid"

    def __init__(self, fluid_stencil, thermal_stencil, shape: Tuple[int]):
        super().__init__(
            [
                FluidLattice(fluid_stencil, shape),
                FluidLattice(thermal_stencil, shape),
            ]
        )

    @partial(jit, static_argnums=(0))
    def initialize(
        self,
        density: ArrayLike,
        temperature: ArrayLike,
        velocity: ArrayLike,
    ) -> Array:
        """Initialize the distribution function with the equilibrium distribution
        corresponding to the given density, temperature and velocity.

        Args:
            density (ArrayLike): The prescribed density of the fluid.
            temperature (ArrayLike): The prescribed temperature of the fluid.
            velocity (ArrayLike): The prescribed velocity of the fluid.

        Returns:
            Array: The initialized distribution function.
        """

        df_equilibrium = self.equilibrium(
            density=density, velocity=velocity, temperature=temperature
        )
        return df_equilibrium

    @partial(jit, static_argnums=(0))
    def equilibrium(
        self,
        density: ArrayLike,
        velocity: ArrayLike,
        temperature: ArrayLike,
    ) -> Tuple[Array]:
        """Computes the equilibrium distribution functions of the coupled lattixes

        Args:
            density (ArrayLike): The density of the fluid.
            velocity (ArrayLike): The velocity of the fluid.
            temperature (ArrayLike): The temperature of the fluid.

        Returns:
            Tuple[Array]: The equilibrium distribution function for the fluid and
                thermal lattices.
        """

        fluid_eq = self[0].equilibrium(
            m=density,
            u=velocity,
        )

        thermal_eq = self[1].equilibrium(
            m=temperature,
            u=velocity,
        )

        return [fluid_eq, thermal_eq]

    @partial(jit, static_argnums=(0))
    def force(
        self,
        temperature: ArrayLike,
        density: ArrayLike,
        gravity: ArrayLike,
        thermal_expansion: float,
    ) -> Tuple[Array]:
        """Computes the force on the fluid due to gravity.

        Args:
            temperature: The temperature of the fluid.
            density: The density of the fluid.
            gravity: The gravitational acceleration vector.
            thermal_expansion: The thermal expansion coefficient of the fluid.

        Returns:
            The force on the fluid due to gravity.
        """
        # Get the stencil from the first element
        stencil = self[0].stencil

        # Project the gravity vector onto the lattice directions
        e_dot_f = jnp.einsum("dQ, d->Q", stencil.e, gravity)[
            jnp.newaxis, jnp.newaxis, :
        ]

        scalar = 1.0 / stencil.cs**2 * thermal_expansion

        # Compute the force
        fluid_force = (
            scalar
            * density
            * temperature
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
    ) -> List[Tuple]:
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
        density, velocity, temperature = self.get_macroscopics(dist_functions)

        # Compute the equilibrium terms
        equilibrium = self.equilibrium(
            density=density,
            velocity=velocity,
            temperature=temperature,
        )

        # Compute the force terms
        force = self.force(
            temperature=temperature,
            density=density,
            gravity=gravity,
            thermal_expansion=thermal_expansion,
        )

        return equilibrium, force

    @partial(jit, static_argnums=(0))
    def get_macroscopics(self, dist_functions: List[ArrayLike]) -> List[Array]:
        """Get the macroscopic quantities for the coupled NSE and ADE lattices.

        Args:
            dist_functions (List[ArrayLike]): The distribution functions of the
                fluid and thermal lattices.

        Returns:
            List[Array]: The macroscopic quantities for the NSE and ADE lattices.
        """
        density = self[0].get_moment(dist_functions[0], order=0)[..., jnp.newaxis]
        velocity = self[0].get_moment(dist_functions[0], order=1) / density
        temperature = self[1].get_moment(dist_functions[1], order=0)[..., jnp.newaxis]

        return [density, velocity, temperature]
