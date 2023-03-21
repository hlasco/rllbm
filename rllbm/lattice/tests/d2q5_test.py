from rllbm.lattice import D2Q5
import jax.numpy as jnp

def test_D2Q5_moment_0():
    df = jnp.ones((16,16,5))
    m0 = D2Q5.get_moment(df, order=0)
    
    expected = 5
    assert (m0 == expected).all()
    
def test_D2Q5_moment_1():
    df = jnp.ones((16,16,5))
    m1 = D2Q5.get_moment(df, order=1)
    
    expected = jnp.array(
        [0., 0.]
    )
    assert (m1 == expected).all()
    
def test_D2Q5_moment_2():
    df = jnp.ones((16,16,5))
    m2 = D2Q5.get_moment(df, order=2)
    expected = jnp.array(
        [
            [2., 0.],
            [0., 2.],
        ]
    )
    assert (m2 == expected).all()