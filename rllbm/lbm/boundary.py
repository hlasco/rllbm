import abc
from functools import partial
from typing import List, Union, overload, Tuple

from jax import Array, jit
from jax import numpy as jnp
from jax.typing import ArrayLike

from rllbm.lbm.lattice import Stencil, Lattice, CoupledLattices
import jax

__all__ = [
    "BounceBackBoundary",
    "InletBoundary",
    "OutletBoundary",
    "BoundaryDict",
    "apply_boundary_conditions",
]


class Boundary(abc.ABC):
    _mask: ArrayLike
    _nane: str
    _size: int

    def __init__(self, name: str, mask: ArrayLike) -> None:
        self._name = name
        self._mask = mask
        self._size = jnp.sum(mask, dtype=jnp.int32)

    @abc.abstractmethod
    def __call__(
        self, lattice: Lattice, dist_function: ArrayLike, *args, **kwargs
    ) -> ArrayLike:
        """Apply the boundary condition to the given distribution function.

        Args:
            lattice (Lattice): The lattice.
            dist_function (ArrayLike): The distribution function.

        Returns:
            ArrayLike: The distribution function after the application of the
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

    def __init__(self, boundary: Union[Boundary, List[Boundary]] = None):
        self._boundary_dict = {}
        if boundary is not None:
            self.add(boundary)

    def add(self, boundary: Union[Boundary, List[Boundary]]) -> None:
        """Add a boundary to the list of boundaries.

        Args:
            boundary (Union[Boundary, List[Boundary]]): The boundaries to add.
        """
        if isinstance(boundary, list):
            for bdy in boundary:
                if bdy.name in self._boundary_dict:
                    raise ValueError(f"Boundary {bdy.name} already exists.")
                self._boundary_dict[bdy.name] = bdy
        else:
            if boundary.name in self._boundary_dict:
                raise ValueError(f"Boundary {boundary.name} already exists.")
            self._boundary_dict[boundary.name] = boundary

    @partial(jit, static_argnums=(0, 1))
    def __call__(self, lattice: Lattice, dist_function: ArrayLike, **kwargs) -> Array:
        """Apply all the boundary conditions to the given distribution function.

        Args:
            lattice (Lattice): The lattice.
            dist_function (ArrayLike): The distribution function.
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
    def collision_mask(self) -> Array:
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
    def stream_mask(self) -> Array:
        """Return the mask of fluid nodes on which the streaming step is performed. The
        mask corresponds to the logical AND of the stream masks of all the boundaries.

        Returns:
            Array: The stream mask.
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

    def __init__(self, name: str, mask: ArrayLike) -> None:
        super().__init__(name, mask)

    @partial(jit, static_argnums=(0, 1))
    def __call__(
        self,
        lattice: Lattice,
        dist_function: ArrayLike,
    ) -> Array:
        """Apply the bounce back boundary condition to the given distribution function.

        Args:
            lattice (Lattice): The lattice.
            dist_function (ArrayLike): The distribution function.

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

    def __init__(self, name, mask: ArrayLike) -> None:
        super().__init__(name, mask)

    @partial(jit, static_argnums=(0, 1))
    def __call__(
        self,
        lattice: Lattice,
        dist_function: ArrayLike,
        m: Union[ArrayLike, float] = 1.0,
        u: ArrayLike = jnp.array([0.0, 0.0]),
    ) -> Array:
        """Apply the inlet boundary condition to the given distribution function.
        Args:
            lattice (Lattice): The lattice.
            dist_function (ArrayLike): The distribution function.
            m (ArrayLike): The mean of the distribution function.
            u (ArrayLike): The velocity of the distribution function.

        Returns:
            Array: The distribution function after the application of the boundary
                condition.
        """
        if isinstance(m, (int, float)):
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

    def __init__(self, name, mask: ArrayLike, direction: Tuple[int]) -> None:
        self.direction = jnp.array(direction, dtype=jnp.int8)
        self.neighbor_mask = jnp.roll(mask, shift=-self.direction, axis=(0, 1))
        super().__init__(name, mask)

    @partial(jit, static_argnums=(0, 1))
    def __call__(
        self,
        lattice: Lattice,
        dist_function: ArrayLike,
    ) -> Array:
        """Apply the outlet boundary condition to the given distribution function.
        Args:
            lattice (Lattice): The lattice.
            dist_function (ArrayLike): The distribution function.

        Returns:
            Array: The distribution function after the application of the boundary
                condition.
        """
        m, u = lattice.get_macroscopics(dist_function[self.neighbor_mask, :])
        dist_ = lattice.equilibrium(m, u)
        for i in range(lattice.Q):
            dist_function = dist_function.at[self._mask, i].set(dist_[:, i])
        return dist_function

    @property
    def collision_mask(self):
        # Collision is performed on all nodes.
        return ~self._mask

    @property
    def stream_mask(self):
        # Streaming is performed on all nodes except the boundary nodes.
        return ~self._mask


@partial(jit, static_argnums=(0, 1))
def apply_boundary_conditions(
    lattice: Union[Lattice, CoupledLattices],
    boundary_dict: Union[BoundaryDict, List[BoundaryDict]],
    dist_function: Union[ArrayLike, List[ArrayLike]],
    bc_kwargs: Union[dict, List[dict]],
) -> Union[Array, List[Array]]:
    """Apply all the boundary conditions to the given distribution function.

    Args:
        lattice (Lattice): The lattice.
        dist_function (ArrayLike): The distribution function.
        boundary_dict (BoundaryDict): The dictionary of boundaries.
        **kwargs: Additional keyword arguments passed to the boundary conditions.

    Returns:
        Array: The distribution function after the application of the boundary
            conditions.
    """
    if isinstance(lattice, Lattice):
        return boundary_dict(lattice, dist_function, **bc_kwargs)
    else:
        return [
            bdy(l, df, **kw)
            for l, bdy, df, kw in zip(lattice, boundary_dict, dist_function, bc_kwargs)
        ]
