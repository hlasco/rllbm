

from .lattice import Lattice
from .d2q5 import D2Q5
from .d2q9 import D2Q9
from .stream import stream
from .collide import collide

from jax import tree_util
tree_util.register_pytree_node(
    D2Q5,
    D2Q5._tree_flatten,
    D2Q5._tree_unflatten
)

tree_util.register_pytree_node(
    D2Q9,
    D2Q9._tree_flatten,
    D2Q9._tree_unflatten
)