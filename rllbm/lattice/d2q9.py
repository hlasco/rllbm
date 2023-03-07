from rllbm.lattice import Lattice
import jax.numpy as jnp

class D2Q9(Lattice):
    r"""
    LBM Grid: D2Q9
        6   2   5
          \ | /
        3 - 0 - 1
          / | \
        7   4   8 
    """
    def __init__(self):
        dim = 2
        name = "D2Q9"

        coords = jnp.array(
            [
                [ 0, 1, 0, -1,  0, 1, -1, -1,  1,],
                [ 0, 0, 1,  0, -1, 1,  1, -1, -1,],
            ],
        )

        weights = jnp.array(
            [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36,]
        )

        super().__init__(name=name, dim=dim, coords=coords, weights=weights)
        
    @classmethod
    def _tree_unflatten(cls, *args):
        return cls()