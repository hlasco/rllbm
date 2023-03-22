from rllbm.lbm import D2Q9
import jax.numpy as jnp
    
def test_D2Q9_moment_0():
    df = jnp.ones((16,16,9))
    m0 = D2Q9.get_moment(df, order=0)
    
    expected = 9
    assert (m0 == expected).all()
    
def test_D2Q9_moment_1():
    df = jnp.ones((16,16,9))
    m1 = D2Q9.get_moment(df, order=1)

    expected = jnp.array(
        [0., 0.]
    )
    assert (m1 == expected).all()
    
def test_D2Q9_moment_2():
    df = jnp.ones((16,16,9))
    m2 = D2Q9.get_moment(df, order=2)
    
    expected = jnp.array(
        [
            [6., 0.],
            [0., 6.],
        ]
    )

    assert (m2 == expected).all()