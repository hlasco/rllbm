from rllbm.lattice import Lattice
import jax.numpy as jnp

class D2Q5(Lattice):
    r"""
    LBM Grid: D2Q5
            2
            |
        3 - 0 - 1
            |
            4 
    """
    def __init__(self,):
        dim = 2
        name = "D2Q5"

        coords = jnp.array(
            [
                [ 0,  1,  0, -1,  0,],
                [ 0,  0,  1,  0, -1,],
            ],
        )
        
        opposite_indices = jnp.array(
            [
                0, 3, 4, 1, 2,
            ],
        )

        weights = jnp.array(
            [
                1/3, 1/6, 1/6, 1/6, 1/6,
            ]
        )

        super().__init__(
            coords,
            weights,
            opposite_indices,
            name,
            dim
        )