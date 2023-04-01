import abc
from collections.abc import Iterable

from functools import partial
from multipledispatch import dispatch

from typing import Union, Sequence

import chex
import jax

from jax import numpy as jnp

from rllbm.lbm.lattice import Lattice, CoupledLattices

__all__ = [
    "BounceBackBoundary",
    "InletBoundary",
    "OutletBoundary",
    "BoundaryDict",
    "apply_boundary_conditions",
]


class Boundary(abc.ABC):
    _name: str
    _mask: chex.Array
    _size: int

    def __init__(self, name: str, mask: chex.Array) -> None:
        self._name = name
        self._mask = mask
        self._size = jnp.sum(mask, dtype=jnp.int32)

    @abc.abstractmethod
    def __call__(
        self, lattice: Lattice, dist_function: chex.Array, *args, **kwargs
    ) -> chex.Array:
        """Apply the boundary condition to the given distribution function.

        Args:
            lattice (Lattice): The lattice.
            dist_function (chex.Array): The distribution function.

        Returns:
            chex.Array: The distribution function after the application of the
                boundary condition.
        """
        pass

    @property
    def name(self):
        return self._name

    @property
    @abc.abstractmethod
    def collision_mask(self):
        """
        Return the mask of fluid nodes on which the collision step is performed.
        """
        pass

    @property
    @abc.abstractmethod
    def stream_mask(self):
        """
        Return the mask of fluid nodes on which the streaming step is performed.
        """
        pass


class BoundaryDict:
    _boundary_dict: dict

    def __init__(self):
        self._boundary_dict = {}

    @dispatch(Boundary)
    def add(self, boundary: Boundary) -> None:
        """Add a boundary to the list of boundaries.

        Args:
            boundary (Union[Boundary, Sequence[Boundary]]): The boundaries to add.
        """
        if boundary.name in self._boundary_dict:
            raise ValueError(f"Boundary {boundary.name} already exists.")
        self._boundary_dict[boundary.name] = boundary
            
    @dispatch(Iterable)
    def add(self, boundary: Sequence[Boundary]) -> None:
        """Add a boundary to the list of boundaries.

        Args:
            boundary (Sequence[Boundary]): The boundaries to add.
        """
        for bdy in boundary:
            if bdy.name in self._boundary_dict:
                raise ValueError(f"Boundary {bdy.name} already exists.")
            self._boundary_dict[bdy.name] = bdy

    @partial(jax.jit, static_argnums=(0, 1))
    def __call__(self, lattice: Lattice, dist_function: chex.Array, **kwargs) -> chex.Array:
        """Apply all the boundary conditions to the given distribution function.

        Args:
            lattice (Lattice): The lattice.
            dist_function (chex.Array): The distribution function.
            **kwargs: Additional keyword arguments passed to the boundary conditions.

        Returns:
            Array: The distribution function after the application of the boundary
                conditions.
        """
        for name, bdy in self._boundary_dict.items():
            if name in kwargs:
                dist_function = bdy(lattice, dist_function, **kwargs[name])
            else:
                dist_function = bdy(lattice, dist_function)
        return dist_function

    @property
    def collision_mask(self) -> chex.Array:
        """Return the mask of fluid nodes on which the collision step is performed. The
        mask corresponds to the logical AND of the collision masks of all the
        boundaries.

        Returns:
            Array: The collision mask.
        """
        masks = jnp.array(
            [bdy.collision_mask for bdy in self._boundary_dict.values()], dtype=bool
        )
        return jnp.prod(masks, dtype=bool, axis=0)

    @property
    def stream_mask(self) -> chex.Array:
        """Return the mask of fluid nodes on which the streaming step is performed. The
        mask corresponds to the logical AND of the stream masks of all the boundaries.

        Returns:
            chex.Array: The stream mask.
        """
        masks = jnp.array(
            [b.stream_mask for b in self._boundary_dict.values()], dtype=bool
        )
        return jnp.prod(masks, dtype=bool, axis=0)


class BounceBackBoundary(Boundary):
    """A bounce back boundary condition. The distribution function is set to the
    opposite distribution function on the boundary nodes. It can be used to model a
    solid wall.
    """

    def __init__(self, name: str, mask: chex.Array) -> None:
        super().__init__(name, mask)

    @partial(jax.jit, static_argnums=(0, 1))
    def __call__(
        self,
        lattice: Lattice,
        dist_function: chex.Array,
    ) -> chex.Array:
        """Apply the bounce back boundary condition to the given distribution function.

        Args:
            lattice (Lattice): The lattice.
            dist_function (chex.Array): The distribution function.

        Returns:
            Array: The distribution function after the application of the boundary
                condition.
        """
        for i in range(lattice.Q):
            dist_function = dist_function.at[self._mask, i].set(
                dist_function[self._mask, lattice.stencil.opposite[i]],
            )
        return dist_function

    @property
    def collision_mask(self):
        # Collision is performed on all nodes except the boundary nodes.
        return ~self._mask

    @property
    def stream_mask(self):
        # Streaming is performed on all nodes except the boundary nodes.
        return ~self._mask


class InletBoundary(Boundary):
    """An inlet boundary condition. The distribution function is set to the
    equilibrium distribution function on the boundary nodes. It can be used to model
    an inlet.
    """

    def __init__(self, name, mask: chex.Array) -> None:
        super().__init__(name, mask)

    @partial(jax.jit, static_argnums=(0, 1))
    def __call__(
        self,
        lattice: Lattice,
        dist_function: chex.Array,
        m: Union[chex.Array, chex.Scalar] = 1.0,
        u: chex.Array = jnp.array([0.0, 0.0]),
    ) -> chex.Array:
        """Apply the inlet boundary condition to the given distribution function.
        Args:
            lattice (Lattice): The lattice.
            dist_function (chex.Array): The distribution function.
            m (chex.Array): The mean of the distribution function.
            u (chex.Array): The velocity of the distribution function.

        Returns:
            Array: The distribution function after the application of the boundary
                condition.
        """
        if isinstance(m, chex.Scalar):
            m = m * jnp.ones((self._size))
        m = m[..., jnp.newaxis]

        if u.ndim == 1:
            u = jnp.ones((self._size, lattice.D)) * u[jnp.newaxis, :]

        equilibrium = lattice.equilibrium(m, u)

        for i in range(lattice.Q):
            dist_function = dist_function.at[self._mask, i].set(equilibrium[:, i])
        return dist_function

    @property
    def collision_mask(self):
        # Collision is performed on all nodes
        return jnp.ones_like(~self._mask)

    @property
    def stream_mask(self):
        # Streaming is performed on all nodes except the boundary nodes.
        return ~self._mask


class OutletBoundary(Boundary):
    """An outlet boundary condition."""

    def __init__(self, name, mask: chex.Array, direction: Sequence[int]) -> None:
        self.direction = jnp.array(direction, dtype=jnp.int8)
        self.neighbor_mask = jnp.roll(mask, shift=-self.direction, axis=(0, 1))
        super().__init__(name, mask)

    @partial(jax.jit, static_argnums=(0, 1))
    def __call__(self, lattice: Lattice, dist_function: chex.Array) -> chex.Array:
        """Apply the outlet boundary condition to the given distribution function.
        Args:
            lattice (Lattice): The lattice.
            dist_function (chex.Array): The distribution function.

        Returns:
            Array: The distribution function after the application of the boundary
                condition.
        """
        fluid_state = lattice.get_macroscopics(dist_function[self.neighbor_mask, :])
        eq = lattice.equilibrium(fluid_state)
        for i in range(lattice.Q):
            dist_function = dist_function.at[self._mask, i].set(eq[:, i])
        return dist_function

    @property
    def collision_mask(self):
        # Collision is performed on all nodes.
        return ~self._mask

    @property
    def stream_mask(self):
        # Streaming is performed on all nodes except the boundary nodes.
        return ~self._mask

@dispatch(Lattice, BoundaryDict, jax.Array, dict)
@partial(jax.jit, static_argnums=(0, 1))
def apply_boundary_conditions(
    lattice: Lattice,
    boundary_dict: BoundaryDict,
    dist_function: chex.Array,
    bc_kwargs: dict,
) -> chex.Array:
    """Apply all the boundary conditions to the given distribution function.

    Args:
        lattice (Lattice): The lattice.
        boundary_dict (BoundaryDict): The dictionary of boundaries.
        dist_function (chex.Array): The distribution function.
        dist_function (chex.Array): The distribution function.
        bc_kwargs: Additional keyword arguments passed to the boundary conditions.

    Returns:
        Array: The distribution function after the application of the boundary
            conditions.
    """
    return boundary_dict(lattice, dist_function, **bc_kwargs)

@dispatch(CoupledLattices, Iterable, Iterable, Iterable)
@partial(jax.jit, static_argnums=(0, 1))
def apply_boundary_conditions(
    lattice: CoupledLattices,
    boundary_dict: Sequence[BoundaryDict],
    dist_function: Sequence[chex.Array],
    bc_kwargs: Sequence[dict],
) -> Sequence[chex.Array]:
    """Apply all the boundary conditions to the given distribution function.

    Args:
        lattice (CoupledLattices): The lattice.
        dist_function (Sequence[chex.Array]): The distribution function.
        dist_function (Sequence[chex.Array]): The distribution function.
        boundary_dict (Sequence[BoundaryDict]): The dictionary of boundaries.
        bc_kwargs (Sequence[dict]): Additional keyword arguments passed to the boundary conditions.

    Returns:
        Array: The distribution function after the application of the boundary
            conditions.
    """
    return [
        bdy(l, df, **kw)
        for l, bdy, df, kw in zip(lattice.to_tuple(), boundary_dict, dist_function, bc_kwargs)
    ]