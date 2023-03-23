import jax.numpy as jnp
import jax

from rllbm import lbm as lbm

from tqdm.rich import trange
from tqdm import TqdmExperimentalWarning
import netCDF4

import warnings

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

@jax.jit
def get_wall_temperature(t, x, freq_x0=0.005, freq_amp=0.02, amp=0.5, width=0.1):
    """
    Function to calculate the bottom temperature at the current time step (t) and spatial location (y)
        nx is the number of grid points in the x-direction
        freq_x0 is the frequency of the sinusoidal variation in the x-direction
        freq_amp is the frequency of the sinusoidal variation in the amplitude
        amp is the maximum amplitude of the temperature
        width is the width of the temperature profile
    """
    x_min = x[0]
    x_max = x[-1]
    
    x_ = (x - x_min) / (x_max - x_min)

    x_peak = 0.5 * ( 1 + jnp.sin(2 * jnp.pi * t * freq_x0))
    amp = amp * jnp.sin(x_  * jnp.pi) * jnp.cos(2 * jnp.pi * t * freq_amp)
    ret = amp * jnp.exp(-0.5 * ((x_ - x_peak) / (width)) ** 2)
    return ret


def init_ncfile(path, sim):
    """Creates a new netCDF4 file, using the specified path and simulation."""
    with netCDF4.Dataset(path, "w", format="NETCDF4") as ncfile:
        ncfile.createDimension("nx", sim.nx)
        ncfile.createDimension("ny", sim.ny)
        ncfile.createDimension("time", None)

        ncfile.createVariable("x", "f4", ("nx",))
        ncfile.createVariable("y", "f4", ("ny",))
        ncfile.createVariable("t", "f4", ("time",))

        ncfile.coordinates = "t x y"

        ncfile.createVariable(
            "temp",
            "f4",
            ("time", "nx", "ny"),
            compression="zlib",
            least_significant_digit=3,
        ),

        ncfile.variables["x"][:] = sim.x * sim.dx
        ncfile.variables["y"][:] = sim.y * sim.dx


def write_ncfile(path, time_index, time, temperature):
    """Write the temperature data to a netCDF file.
    """
    with netCDF4.Dataset(path, "a") as ncfile:
        ncfile.variables["t"][time_index] = time
        ncfile.variables["temp"][time_index, :, :] = temperature[:, :, 0]

if __name__ == "__main__":
    nc_path = "outputs.nc"

    # Simulation parameters
    nx = 128
    ny = 128

    prandtl = 0.71
    rayleigh = 1e9
    thermal_expansion = 0.0005
    gravity = 9.81
    buoyancy = gravity * thermal_expansion

    dx = 1.0 / (max(nx, ny))
    dt = (buoyancy * dx) ** 0.5
    
    # The run time in code units
    run_time = 100.0
    # Number of steps to run
    steps = int(run_time / dt)
    # Frequency of writing to the netCDF file
    io_frequency = int(1.0 / dt)

    # Collision parameters
    viscosity = (prandtl / rayleigh) ** 0.5 * dt / dx**2
    kappa = 1.0 / (prandtl * rayleigh) ** 0.5 * dt / dx**2
    omegas = (
        1.0 / (3 * viscosity + 0.5),
        1.0 / (3 * kappa + 0.5),
    )

    # ConvectionLattice needs to know those parameters in order to calculate the
    # collision terms
    collision_kwargs = {
        "gravity": jnp.array([0, gravity]),
        "thermal_expansion": thermal_expansion,
    }

    # Instantiate the simulation
    sim = lbm.Simulation(nx, ny, dt, omegas, collision_kwargs)

    X, Y = jnp.meshgrid(sim.x, sim.y, indexing="ij")

    # Instantiate the lattice
    lattice = lbm.ConvectionLattice(
        fluid_stencil=lbm.D2Q9,
        thermal_stencil=lbm.D2Q5,
        shape=(nx, ny),
    )

    # Initialize the density functions
    dfs = lattice.initialize(
        density=jnp.ones((nx, ny, 1)),
        velocity=jnp.zeros((nx, ny, 2)),
        temperature=jnp.zeros((nx, ny, 1)),
    )
    sim.initialize(lattice, dfs)

    # Set the boundary conditions
    fluid_bc = lbm.BoundaryDict(
        [
            lbm.BounceBackBoundary(
                "No-Slip Walls", (X == 0) | (X == nx - 1) | (Y == 0) | (Y == ny - 1)
            ),
        ]
    )
    thermal_bc = lbm.BoundaryDict(
        [
            lbm.BounceBackBoundary("No-Slip Walls", (Y == 0) | (Y == ny - 1)),
            lbm.InletBoundary("Left Wall", (X == 0)),
            lbm.InletBoundary("Right Wall", (X == nx - 1)),
        ]
    )
    fluid_bc_kwargs = {}
    
    thermal_bc_kwargs = {
        "Left Wall": {"m": get_wall_temperature(0, sim.y)},
        "Right Wall": {"m": 0.0},
    }
    
    sim.set_boundary_conditions(
        (fluid_bc, thermal_bc),
        (fluid_bc_kwargs, thermal_bc_kwargs),
    )
    
    init_ncfile(nc_path, sim)

    for i in trange(steps):
        sim.step()
        
        # Update the left wall temperature
        sim.bc_kwargs[1]["Left Wall"]["m"] = get_wall_temperature(sim.time, sim.y)

        if i % io_frequency == 0:
            
            _, velocity, temperature = sim.get_macroscopics(sim.dfs)
            
            if jnp.isnan(temperature).any():
                print("NaNs detected, stopping simulation.")
                break

            time_index = i // io_frequency
            write_ncfile(nc_path, time_index, sim.time, temperature)
