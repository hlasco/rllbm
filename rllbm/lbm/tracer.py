from collections import defaultdict
from collections.abc import Iterable
from functools import partial

from typing import List, Union
import chex
import jax
from jax import numpy as jnp

import numpy as np

from rllbm.lbm import Simulation

__all__ = ["Tracer", "TracerCollection", "update_tracers"]

@chex.dataclass
class Tracer:
    stream: bool
    x: chex.Array
    obs: chex.Array = None
    
    @classmethod
    def random(cls, key: jax.random.PRNGKey, sim: Simulation, stream: bool = True):
        """Creates a tracer with a random position in the simulation domain

        Args:
            sim (Simulation): The simulation object
            stream (bool, optional): Whether the tracer should be streamed. Defaults to True.

        Returns:
            Tracer: The tracer object
        """
        minval = jnp.array([_min+sim.dx for _min in sim.bounds[::sim.dim]])
        maxval = jnp.array([_max-sim.dx for _max in sim.bounds[1:][::sim.dim]])
        x = jax.random.uniform(key, shape=(sim.dim,), minval=minval, maxval=maxval)
        return cls(stream=stream, x=x, obs=None)
    
    def update_x(self, x):
        self.x = x
    
    def update_obs(self, obs):
        self.obs = obs

class TracerCollection:
    def __init__(self, tracers=None):
        if tracers is None:
            self.tracers = defaultdict(list)
        else:
            self.tracers = tracers
            
        self.index_map = self._build_index_map()

    def _build_index_map(self):
        index_map = {}
        for tracer_type, tracer_list in self.tracers.items():
            for pos in range(len(tracer_list)):
                index_map[len(index_map)] = (tracer_type, pos)
        return index_map
    
    def __len__(self):
        return len(self.index_map)

    def __iter__(self):
        for tracer_list in self.tracers.values():
            for tracer in tracer_list:
                yield tracer
                
    def __getitem__(self, idx):
        if not isinstance(idx, int):
            raise TypeError("Index must be an integer")
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        tracer_type, pos = self.index_map[idx]
        return self.tracers[tracer_type][pos]
    
    def __setitem__(self, index, tracer: Tracer):
        if not isinstance(tracer, Tracer):
            raise TypeError("Tracer must be of type Tracer")
        if not isinstance(index, int):
            raise TypeError('Invalid index type')
        
        tracer_type, pos = self.index_map[index]
        if pos >= len(self.tracers[tracer_type]):
            raise IndexError('Tracer index out of range')
        self.tracers[tracer_type][pos] = tracer
            

    def add(self, tracer_type: str, tracer: Tracer):
        """Adds a tracer to the tracer collection

        Args:
            tracer_type (str): The type of tracer
            tracer (Tracer): The tracer to add
        """
        self.tracers[tracer_type].append(tracer)
        idx = len(self.index_map)
        self.index_map[idx] = (tracer_type, len(self.tracers[tracer_type])-1)
        
    def add_random_tracers(
        self, key: jax.random.PRNGKey, tracer_type: str, sim: Simulation, num: int, stream: bool = True
    ):
        """Adds n random tracers to the tracer collection

        Args:
            key (jax.random.PRNGKey): Random key
            tracer_type (str): The type of tracer
            sim (Simulation): The simulation object
            num (int): The number of tracers to add
            stream (bool, optional): Whether the tracers should be streamed. Defaults to True.
        """
        keys = jax.random.split(key, num)
        for i in range(num):
            tracer = Tracer.random(keys[i], sim, stream)
            self.add(tracer_type, tracer)

    def add_grid_tracers(self, tracer_type: str, sim: Simulation, shape: Iterable[int], stream: bool = True):
        """Adds a grid of tracers to the tracer collection

        Args:
            tracer_type (str): The type of tracer.
            sim (Simulation): The simulation object.
            shape (Iterable[int]): The shape of the grid (e.g. (10, 10) for a 10x10 grid).
            stream (bool, optional): Whether the tracers should be streamed. Defaults to True.
        """
        if len(shape) != sim.dim:
            raise ValueError(f"Shape must be of length {sim.dim}")

        linspaces = []
        for i in range(sim.dim):
            start = sim.bounds[i * 2]+sim.dx
            stop = sim.bounds[i * 2 + 1]-sim.dx
            size = shape[i]
            linspaces.append(jnp.linspace(start, stop, size))
        
        for idx in np.ndindex(shape):
            x = jnp.array([linspaces[i][idx[i]] for i in range(sim.dim)])
            tracer = Tracer(stream=stream, x=x, obs=None)
            self.add(tracer_type, tracer)

    @property
    def types(self):
        return self.tracers.keys()

    def get(self, tracer_type: str):
        return self.tracers[tracer_type]

    def set_tracer(self, tracer_type: str, idx: int, tracer: Tracer):
        self.tracers[tracer_type][idx] = tracer

    def _tree_flatten(self):
        children = (self.tracers,)  # arrays / dynamic values
        aux_data = {}  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


jax.tree_util.register_pytree_node(
    TracerCollection,
    TracerCollection._tree_flatten,
    TracerCollection._tree_unflatten
)

@partial(jax.jit, static_argnums=(0))
def update_tracers(
    sim: Simulation, fluid_state: Iterable[chex.Array], tracers: TracerCollection
):
    
    xmin = jnp.array([_min + sim.dx for _min in sim.bounds[::sim.dim]])
    xmax = jnp.array([_max - sim.dx for _max in sim.bounds[1:][::sim.dim]])

    for i, tracer in enumerate(tracers):
        idx = jnp.floor(tracer.x / sim.dx).astype(int)
        
        u = fluid_state.u[idx[0], idx[1]]
        
        x = tracer.x + u * sim.dx * tracer.stream
        x = jnp.clip(x, a_min = xmin, a_max = xmax)
        
        obs = jnp.concatenate([state[tuple(idx)] for state in fluid_state])
        
        tracer.x = x
        tracer.obs = obs
        
        tracers[i] = tracer

    return tracers