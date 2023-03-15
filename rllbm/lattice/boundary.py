import jax.numpy as jnp

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