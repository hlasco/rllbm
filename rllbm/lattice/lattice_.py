
import abc
from jax import Array
import jax.numpy as jnp
from jax.typing import ArrayLike
from typing import Optional, Tuple

from rllbm.lattice import Stencil

__all__ = ["Lattice", "NavierStokesLattice", "AdvectionDiffusionLattice"]

class Lattice(abc.ABC):
    _stencil: Stencil
    _shape: Tuple[int]
    _df: ArrayLike
    _name: str = "BaseLattice"
    
    def __init__(self, stencil: Stencil, shape: Tuple[int]):
        self._stencil = stencil
        self._shape = shape
        self._df = jnp.empty((*shape, stencil.Q))
    
    @property
    def name(self):
        return self._name
    
    @property
    def stencil(self):
        return self._stencil
    
    @property
    def D(self):
        return self._stencil.D
    
    @property
    def Q(self):
        return self._stencil.Q
    
    @property
    def df(self):
        return self._df
    
    @property
    def shape(self):
        return self._shape
    
    def get_moment(self, order: int):
        return self._stencil.get_moment(self._df, order)
    
    def initialize(self, *args, **kwargs) -> Array:
        self._df = self.df_equilibrium(*args, **kwargs)
    
    @abc.abstractmethod
    def df_equilibrium(self, *args, **kwargs) -> Array:
        pass
    
    @abc.abstractmethod
    def df_force(self, *args, **kwargs) -> Array:
        pass
    
class CoupledLattice(abc.ABC):
    def __init__(self, lattices: Tuple[Tuple]):
        self._lattices = dict([l.name, l] for l in lattices)
        
    @abc.abstractmethod
    def df_equilibrium(self):
        pass
    
    @abc.abstractmethod
    def df_force(self, *args, **kwargs) -> Array:
        pass
    
class NavierStokesLattice(Lattice):
    
    def __init__(
        self,
        stencil: Stencil,
        shape: Tuple[int],
    ):
        super().__init__(stencil, shape)
    
    def df_equilibrium(
        self,
        density: Optional[ArrayLike]=None,
        velocity: Optional[ArrayLike]=None
    ) -> Array:

        if not density:
            density = self.get_moment(order=0)
        if not velocity:
            velocity =  self.get_moment(order=1) / density
        
        density = density[..., jnp.newaxis]
            
        assert velocity.shape == (*self._shape, self.D)
        assert density.shape == (*self._shape, 1)
        
        u_norm2 = jnp.linalg.norm(velocity, axis=-1, ord=2,)[..., jnp.newaxis]**2
        
        e_dot_u = jnp.einsum("dQ, NMd->NMQ", self._stencil.e, velocity)
        
        cs = self._stencil.cs
        
        df_eq = density * self._stencil.w[jnp.newaxis, jnp.newaxis, :] * (
            1 + e_dot_u / cs ** 2 + 0.5 * e_dot_u ** 2 / cs ** 4 - 0.5 * u_norm2 / cs ** 2
        )
        
        return df_eq
    
    def df_force(
        self,
        temperature: ArrayLike,
        gravity: ArrayLike,
        thermal_expansion: float,
        dt: float,
        density: Optional[ArrayLike]=None,
    ) -> Array:
        if not density:
            density = self.get_moment(order=0)
        
        density = density[..., jnp.newaxis]
        temperature = temperature[..., jnp.newaxis]
        
        assert temperature.shape == (*self._shape, 1)
        assert density.shape == (*self._shape, 1)
        assert gravity.shape == (self._stencil.D)
        
        e_dot_f = jnp.einsum("dQ, d->Q", self._stencil.e, gravity)[jnp.newaxis, jnp.newaxis, :]
        
        scalar = 1.0 / self._stencil.cs ** 2 * thermal_expansion * dt
        
        force = scalar * density * temperature * e_dot_f * self._stencil.w[jnp.newaxis, jnp.newaxis, :]

        return force
    
class AdvectionDiffusionLattice(Lattice):
    
    def __init__(self, stencil: Stencil, shape: Tuple[int]):
        super().__init__(stencil, shape)
    
    def df_equilibrium(
        self,
        velocity: ArrayLike,
        temperature: Optional[ArrayLike]=None,
    ) -> Array:
        
        if not temperature:
            temperature = self.get_moment(order=0)
        
        temperature = temperature[..., jnp.newaxis]
        
        assert velocity.shape == (*self._shape, self.D)
        assert temperature.shape == (*self._shape, 1)
        
        e_dot_u = jnp.einsum("dQ, NMd->NMQ", self._stencil.e, velocity)
        
        cs = self._stencil.cs
        
        df_eq = temperature * self._stencil.w[jnp.newaxis, jnp.newaxis, :] * (
            1 + e_dot_u / cs ** 2
        )
        
        return df_eq
    
    def df_force(self):
        return jnp.zeros((*self._shape, self.D))