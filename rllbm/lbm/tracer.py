from collections import defaultdict

from typing import List
import chex

from jax import Array
from jax import numpy as jnp
from jax import tree_util

__all__ = ["Tracer", "TracerCollection"]

@chex.dataclass
class Tracer:
    stream: bool
    x: chex.Array
    obs: chex.Array = None

class TracerCollection:
    def __init__(self, tracers=None):
        if tracers is None:
            self.tracers = defaultdict(list)
        else:
            self.tracers = tracers

    def add(self, tracer_type: str, tracer: Tracer):
        self.tracers[tracer_type].append(tracer)

    def keys(self):
        return self.tracers.keys()

    def values(self):
        return self.tracers.values()

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


tree_util.register_pytree_node(
    TracerCollection,
    TracerCollection._tree_flatten,
    TracerCollection._tree_unflatten
)