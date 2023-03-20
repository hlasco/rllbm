import abc
from functools import partial
from typing import List, Union, overload

from jax import Array, jit
from jax import numpy as jnp
from jax.typing import ArrayLike

from rllbm.lattice.lattice import Lattice, CoupledLattices

__all__ = ["BounceBackBoundary", "DirichletBoundary", "BoundaryDict", "apply_boundary_conditions"]

class Boundary(abc.ABC):
    _mask: ArrayLike
    _nane: str
    
    def __init__(self, name: str, mask: ArrayLike) -> None:
        self._name = name
        self._mask = mask
        
    @abc.abstractmethod
    def __call__(
        self,
        lattice: Lattice,
        dist_function: ArrayLike,
        *args,
        **kwargs
    ) -> ArrayLike:
        """Apply all the boundary condition to the given distribution function.

        Args:
            lattice (Lattice): The lattice.
            dist_function (ArrayLike): The distribution function.

        Returns:
            ArrayLike: The distribution function after the application of the boundary condition.
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
    def __call__(
        self,
        lattice: Lattice,
        dist_function: ArrayLike,
        **kwargs
    ) -> Array:
        """Apply all the boundary conditions to the given distribution function.

        Args:
            lattice (Lattice): The lattice.
            dist_function (ArrayLike): The distribution function.
            **kwargs: Additional keyword arguments passed to the boundary conditions.

        Returns:
            Array: The distribution function after the application of the boundary conditions.
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
        mask corresponds to the logical AND of the collision masks of all the boundaries.

        Returns:
            Array: The collision mask.
        """
        masks = jnp.array([bdy.collision_mask for bdy in self._boundary_dict.values()], dtype=bool)
        return jnp.prod(masks, dtype=bool, axis=0)
    
    @property
    def stream_mask(self) -> Array:
        """Return the mask of fluid nodes on which the streaming step is performed. The
        mask corresponds to the logical AND of the stream masks of all the boundaries.

        Returns:
            Array: The stream mask.
        """
        masks = jnp.array([b.stream_mask for b in self._boundary_dict.values()], dtype=bool)
        return jnp.prod(masks, dtype=bool, axis=0)
    
class BounceBackBoundary(Boundary):
    """ A bounce back boundary condition. The distribution function is set to the opposite
    distribution function on the boundary nodes. It can be used to model a solid wall.
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
            Array: The distribution function after the application of the boundary condition.
        """
        for i in range(lattice.Q):
            dist_function = dist_function.at[self._mask,i].set(
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
    
class DirichletBoundary(Boundary):
    """ A Dirichlet boundary condition. The distribution function is set so that the macroscopic
    value of the fluid is equal to the given value on the boundary nodes. It can be used to model a
    fluid inlet or outlet with a given value of a macroscopic variable.
    """
    
    def __init__(self, name, mask: ArrayLike) -> None:
        super().__init__(name, mask)
    
    @partial(jit, static_argnums=(0, 1))
    def __call__(
        self,
        lattice: Lattice,
        dist_funcion: ArrayLike,
        fixed_value: ArrayLike = 0.,
    ) -> Array:
        """Apply the dirichlet boundary condition to the given distribution function.
        Args:
            lattice (Lattice): The lattice.
            dist_funcion (ArrayLike): The distribution function.
            fixed_value (ArrayLike): The value of the macroscopic variable on the boundary.
        
        Returns:
            Array: The distribution function after the application of the boundary condition.
        """
        for i in range(lattice.Q):
            dist_funcion = dist_funcion.at[self._mask,i].set(
                1. / lattice.Q * fixed_value
            )
        return dist_funcion
    
    @property
    def collision_mask(self):
        # Collision is performed on all nodes except the boundary nodes.
        return ~self._mask
    
    @property
    def stream_mask(self):
        # Streaming is performed on all nodes.
        return jnp.ones_like(self._mask)

@overload
@partial(jit, static_argnums=(0, 1))
def apply_boundary_conditions(
    lattice: Lattice,
    boundary_dict: BoundaryDict,
    dist_function: ArrayLike,
    **bc_kwargs
) -> Array:
    """Apply all the boundary conditions to the given distribution function.

    Args:
        lattice (Lattice): The lattice.
        dist_function (ArrayLike): The distribution function.
        boundary_dict (BoundaryDict): The dictionary of boundaries.
        **kwargs: Additional keyword arguments passed to the boundary conditions.

    Returns:
        Array: The distribution function after the application of the boundary conditions.
    """
    return boundary_dict(lattice, dist_function, **bc_kwargs)


@partial(jit, static_argnums=(0, 1))
def apply_boundary_conditions(
    lattices: CoupledLattices,
    boundary_dicts: List[BoundaryDict],
    dist_functions: List[ArrayLike],
    **bc_kwargs,
) -> List[Array]:
    """Apply all the boundary conditions to the given distribution functions.

    Args:
        lattice (CoupledLattices): The coupled lattices.
        dist_functions (List[ArrayLike]): The distribution functions.
        boundary_dicts (List[BoundaryDict]): The dictionaries of boundaries.
        **bc_kwargs: Additional keyword arguments passed to the boundary conditions.

    Returns:
        List[Array]: The distribution functions after the application of the boundary conditions.
    """
    return [
        boundary_dict(lattice, dist_function, **bc_kwargs)
        for lattice, boundary_dict, dist_function in zip(lattices, boundary_dicts, dist_functions)
    ]
