import abc
from functools import partial
from typing import List, Union, overload

from jax import Array, jit
from jax import numpy as jnp
from jax.typing import ArrayLike

from rllbm.lbm.lattice import Lattice, CoupledLattices

__all__ = [
    "BounceBackBoundary",
    "InletBoundary",
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
        """Apply all the boundary condition to the given distribution function.

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
        dist_funcion: ArrayLike,
        m: Union[ArrayLike, float] = 0.0,
        u: ArrayLike = jnp.array([0.0, 0.0]),
    ) -> Array:
        """Apply the dirichlet boundary condition to the given distribution function.
        Args:
            lattice (Lattice): The lattice.
            dist_funcion (ArrayLike): The distribution function.
            m (ArrayLike): The first moment of the distribution function.
            u (ArrayLike): The second moment of the distribution function.

        Returns:
            Array: The distribution function after the application of the boundary
                condition.
        """

        if m.shape == ():
            m = m * jnp.ones((self._size))
        m = m[..., jnp.newaxis]

        if u.shape == (2,):
            u = jnp.ones((self._size, 2)) * u[jnp.newaxis, :]

        equilibrium = lattice.equilibrium(m, u)

        for i in range(lattice.Q):
            dist_funcion = dist_funcion.at[self._mask, i].set(equilibrium[:, i])
        return dist_funcion

    @property
    def collision_mask(self):
        # Collision is performed on all nodes except the boundary nodes.
        return ~self._mask

    @property
    def stream_mask(self):
        # Streaming is performed on all nodes except the boundary nodes.
        return ~self._mask


@overload
@partial(jit, static_argnums=(0, 1))
def apply_boundary_conditions(
    lattice: Lattice,
    boundary_dict: BoundaryDict,
    dist_function: ArrayLike,
    **bc_kwargs,
) -> Array:
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
    return boundary_dict(lattice, dist_function, **bc_kwargs)


@partial(jit, static_argnums=(0, 1))
def apply_boundary_conditions(
    lattices: CoupledLattices,
    boundary_dicts: List[BoundaryDict],
    dist_functions: List[ArrayLike],
    bc_kwargs: List[dict],
) -> List[Array]:
    """Apply all the boundary conditions to the given distribution functions.

    Args:
        lattice (CoupledLattices): The coupled lattices.
        dist_functions (List[ArrayLike]): The distribution functions.
        boundary_dicts (List[BoundaryDict]): The dictionaries of boundaries.
        **bc_kwargs: Additional keyword arguments passed to the boundary conditions.

    Returns:
        List[Array]: The distribution functions after the application of the boundary
            conditions.
    """
    return [
        boundary_dict(lattice, dist_function, **kwargs)
        for lattice, boundary_dict, dist_function, kwargs in zip(
            lattices, boundary_dicts, dist_functions, bc_kwargs
        )
    ]
