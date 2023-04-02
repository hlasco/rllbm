import jax.numpy as jnp
import jax

from rllbm import lbm as lbm

from tqdm.rich import trange
from tqdm import TqdmExperimentalWarning
import netCDF4

import warnings

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
import matplotlib.pyplot as plt


@jax.jit
def get_inlet_velocity(x, amp=0.1, direction=0):
    """ """
    x_min = x[0]
    x_max = x[-1]

    x_ = (x - x_min) / (x_max - x_min)

    velocity = jnp.zeros((x.shape[0], 2))
    velocity = velocity.at[:, direction].set(amp * x_ * (1 - x_) / 0.5**2)
    return velocity


def init_ncfile(path, sim):
    """Creates a new netCDF4 file, using the specified path and simulation."""
    with netCDF4.Dataset(path, "w", format="NETCDF4") as ncfile:
        ncfile.createDimension("nx", sim.shape[0])
        ncfile.createDimension("ny", sim.shape[1])
        ncfile.createDimension("time", None)

        ncfile.createVariable("x", "f4", ("nx",))
        ncfile.createVariable("y", "f4", ("ny",))
        ncfile.createVariable("t", "f4", ("time",))

        ncfile.coordinates = "t x y"

        ncfile.createVariable(
            "curl",
            "f4",
            ("time", "nx", "ny"),
            compression="zlib",
            least_significant_digit=2,
        ),

        ncfile.variables["x"][:] = sim.x
        ncfile.variables["y"][:] = sim.y


def write_ncfile(path, time_index, time, curl):
    """Write the temperature data to a netCDF file."""
    with netCDF4.Dataset(path, "a") as ncfile:
        ncfile.variables["t"][time_index] = time
        ncfile.variables["curl"][time_index, :, :] = curl[:, :]


if __name__ == "__main__":
    nc_path = "outputs.nc"

    # Simulation parameters
    nx, ny = 192, 48

    mach = 0.05
    re = 150

    domain = lbm.Domain(shape=(nx, ny), bounds=(0.0, nx / ny, 0.0, 1.0))

    dx = domain.dx
    dt = dx
    obs_x = domain.width[0] / 5
    obs_y = domain.width[1] * 0.4
    obs_size = domain.width[1] / 8

    # strouhal number
    st = 0.198 * (1.0 - 19.7 / re)
    v0 = mach * jnp.sqrt(3)
    vortex_period = 2 * obs_size / (st * v0)

    viscosity = (v0 * obs_size / dx) / re
    omega = 1.0 / (3 * viscosity + 0.5)

    # The run time in code units
    # Simulate 10 periods
    run_time = 20 * vortex_period

    # Number of steps to run
    steps = int(run_time / dt)
    # Frequency of writing to the netCDF file
    io_frequency = int(0.1 * vortex_period / dt)

    # Instantiate the lattice
    lattice = lbm.FluidLattice(lbm.D2Q9)

    sim = lbm.Simulation(domain, lattice, omega)
    sim.set_initial_conditions(
        rho=jnp.ones((nx, ny, 1)),
        u=jnp.zeros((nx, ny, 2)),
    )

    X, Y = jnp.meshgrid(sim.x, sim.y, indexing="ij")
    obstacle_mask = (jnp.fabs(X - obs_x) < obs_size) & (jnp.fabs(Y - obs_y) < obs_size)
    # Set the boundary conditions
    fluid_bc = [
        lbm.InletBoundary("Left", sim.left),
        lbm.OutletBoundary("Right", sim.right, direction=[1, 0]),
        lbm.BounceBackBoundary("Top", sim.top),
        lbm.BounceBackBoundary("Bot", sim.bottom),
        lbm.BounceBackBoundary("Obstacle", obstacle_mask),
    ]

    sim.set_boundary_conditions(fluid_bc)
    inlet_vel = get_inlet_velocity(sim.y, amp=v0, direction=0)
    sim.update_boundary_condition("Left", {"u": inlet_vel})

    init_ncfile(nc_path, sim)

    for i in trange(steps):
        sim.step()
        t = i * dt

        if i % io_frequency == 0:
            fluid_state = sim.get_macroscopics()

            _, dudy = jnp.gradient(fluid_state.u[..., 0], dx)
            dvdx, _ = jnp.gradient(fluid_state.u[..., 1], dx)
            curl = dudy - dvdx

            time_index = i // io_frequency
            write_ncfile(nc_path, time_index, t, curl)

            if jnp.isnan(fluid_state.u).any():
                print("NaNs detected, stopping simulation.")
                break
