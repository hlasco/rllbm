from jax import Array
import jax.numpy as jnp
from jax.typing import ArrayLike
import abc
from typing import Tuple
from rllbm.lattice import Lattice

__all__ = ["Boundary", "BounceBackBoundary", "AntiBounceBackBoundary"]

class Boundary(abc.ABC):
    _mask: ArrayLike
    
    def __init__(self, mask: ArrayLike[bool]) -> None:
        self._mask = mask
        
    @abc.abstractmethod
    def __call__(self, lattice: Lattice, *args, **kwargs) -> Array:
        pass
    
    @property
    @abc.abstractmethod
    def collision_mask(self):
        pass
    
    @property
    @abc.abstractmethod
    def stream_mask(self):
        pass
    
class Boundaries:
    _boundary_list: Tuple[Boundary]
    
    def __init__(self, boundary_list):
        self._boundary_list = boundary_list
        
    def __call__(self, lattice: Lattice) -> Array:
        for boundary in self._boundary_list:
            lattice = boundary(lattice)
        return lattice
    
    @property
    def collision_mask(self):
        return jnp.prod([b.collision_mask for b in self._boundary_list], dtype=bool)
    
    @property
    def stream_mask(self):
        return jnp.prod([b.stream_mask for b in self._boundary_list], dtype=bool)
    
class BounceBackBoundary(Boundary):
    
    def __init__(self, mask: ArrayLike[bool]) -> None:
        super().__init__(mask)
        
    def __call__(self, lattice: Lattice) -> Array:
        for i in range(lattice.Q):
            lattice.df = lattice.df.at[...,i].set(
                jnp.where(
                    self._mask,
                    lattice.df[..., lattice.stencil.opposite[i]],
                    lattice.df[...,i],
                )
            )
        return lattice
    
    @property
    def collision_mask(self):
        return ~self._mask
    
    @property
    def stream_mask(self):
        return ~self._mask
    
class AntiBounceBackBoundary(Boundary):
    def __init__(self, mask: ArrayLike[bool]) -> None:
        super().__init__(mask)
        
    def __call__(self, lattice: Lattice, value: ArrayLike) -> Array:
        for i in range(lattice.Q):
            lattice.df = lattice.df.at[...,i].set(
                jnp.where(
                    self._mask,
                    1. / lattice.Q * value - lattice.df[..., lattice.stencil.opposite[i]],
                    lattice.df[...,i],
                )
            )
        return lattice.df
    
    @property
    def collision_mask(self):
        return jnp.ones_like(self._mask, dtype=bool)
    
    @property
    def no_stream_mask(self):
        return ~self._mask
    

def no_slip_bc(density_function_out, density_function_in, mask_bc, lattice):
    for i in range(lattice.size):
        density_function_out = density_function_out.at[:,:,i].set(
            jnp.where(
                mask_bc,
                density_function_in[:,:, lattice.opposite_indices[i]],
                density_function_out[:,:,i],
            )
        )
    return density_function_out