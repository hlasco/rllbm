import jax.numpy as jnp
import jax

from rllbm import lattice as lbm

from tqdm.rich import trange
from tqdm import TqdmExperimentalWarning
import netCDF4

import warnings

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

@jax.jit
def get_bottom_temperature(t, x, nx, freq_x0=0.005, freq_amp=0.02, amp=0.5, width=0.1):
    """
    Function to calculate the bottom temperature at the current time step (t) and spatial location (x)
        nx is the number of grid points in the x-direction
        freq_x0 is the frequency of the sinusoidal variation in the x-direction
        freq_amp is the frequency of the sinusoidal variation in the amplitude
        amp is the maximum amplitude of the temperature
        width is the width of the temperature profile
    """
    x_0 = 0.5 * (x[-1] - x[0]) * jnp.sin(2 * jnp.pi * t * freq_x0)
    amp = amp * jnp.sin(x / nx * jnp.pi) * jnp.cos(2 * jnp.pi * t * freq_amp)
    return amp * jnp.exp(-0.5 * ((x - nx / 2.0 - x_0) / (width * nx)) ** 2)


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

        ncfile.variables["x"][:] = sim.x / sim.nx
        ncfile.variables["y"][:] = sim.y / sim.ny


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

    dx = 1.0 / (max(nx, ny) - 1.0)
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
        "dt": dt,
    }

    # Instantiate the simulation
    sim = lbm.Simulation(nx, ny, dt, omegas, collision_kwargs)

    X, Y = jnp.meshgrid(sim.x, sim.y, indexing="ij")

    # Instantiate the lattice
    lattice = lbm.ConvectionLattice(
        lbm.NavierStokesLattice(lbm.D2Q9, (nx, ny)),
        lbm.AdvectionDiffusionLattice(lbm.D2Q5, (nx, ny)),
    )

    # Initialize the density functions
    dfs = lattice.initialize(
        density=jnp.ones((nx, ny, 1)),
        velocity=jnp.zeros((nx, ny, 2)),
        temperature=jnp.zeros((nx, ny, 1)),
    )
    sim.initialize(lattice, dfs)

    # Set the boundary conditions
    boundary_NSE = lbm.BoundaryDict(
        [
            lbm.BounceBackBoundary(
                "NSE No-Slip Walls", (X == 0) | (X == nx - 1) | (Y == 0) | (Y == ny - 1)
            ),
        ]
    )
    boundary_ADE = lbm.BoundaryDict(
        [
            lbm.BounceBackBoundary("ADE No-Slip Walls", (Y == 0) | (Y == ny - 1)),
            lbm.DirichletBoundary("ADE Fixed Temperature Bottom Wall", (X == 0)),
            lbm.DirichletBoundary("ADE Fixed Temperature Top Wall", (X == nx - 1)),
        ]
    )
    bc_kwargs = {
        "ADE Fixed Temperature Bottom Wall": {
            "fixed_value": get_bottom_temperature(x=sim.y, t=0, nx=ny),
        },
        "ADE Fixed Temperature Top Wall": {
            "fixed_value": 0.0,
        },
    }
    sim.set_boundary_conditions((boundary_NSE, boundary_ADE), bc_kwargs)
    
    init_ncfile(nc_path, sim)

    for i in trange(steps):
        sim.step()
        
        # Update the bottom wall temperature
        sim.bc_kwargs["ADE Fixed Temperature Bottom Wall"][
            "fixed_value"
        ] = get_bottom_temperature(x=sim.y, t=sim.time, nx=ny)

        if i % io_frequency == 0:
            
            _, velocity, temperature = sim.get_macroscopics(sim.dfs)
            
            if jnp.isnan(temperature).any():
                print("NaNs detected, stopping simulation.")
                break

            time_index = i // io_frequency
            write_ncfile(nc_path, time_index, sim.time, temperature)
