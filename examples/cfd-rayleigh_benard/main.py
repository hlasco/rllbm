import jax.numpy as jnp
import jax

from rllbm import lbm as lbm

from tqdm.rich import trange
from tqdm import TqdmExperimentalWarning
import netCDF4

import warnings

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)


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
            "temp",
            "f4",
            ("time", "nx", "ny"),
            compression="zlib",
            least_significant_digit=4,
        ),

        ncfile.variables["x"][:] = sim.x
        ncfile.variables["y"][:] = sim.y

def write_ncfile(path, time_index, time, state):
    """Write the temperature data to a netCDF file."""
    with netCDF4.Dataset(path, "a") as ncfile:
        ncfile.variables["t"][time_index] = time
        ncfile.variables["temp"][time_index, :, :] = state.T[:, :, 0]


if __name__ == "__main__":
    nc_path = "outputs.nc"

    nx, ny = 64, 64

    domain = lbm.Domain(shape=(nx, ny), bounds=(0.0, 1.0, 0.0, 1.0))

    prandtl = 0.71
    rayleigh = 1e8
    buoyancy = 0.0001

    dx = domain.dx
    dt = (buoyancy * dx) ** 0.5

    # Collision parameters
    viscosity = (prandtl / rayleigh) ** 0.5 * dt / dx**2
    kappa = viscosity / prandtl

    convection_timescale = 1.0
    run_time = 100
    # Number of steps to run
    steps = int(run_time / dt)
    # Frequency of writing to the netCDF file
    io_frequency = int(convection_timescale / dt)

    omegas = {
        "FluidLattice": 1.0 / (3 * viscosity + 0.5),
        "ThermalLattice": 1.0 / (3 * kappa + 0.5),
    }

    # Instantiate the lattice
    lattice = lbm.ThermalFluidLattice(
        fluid_stencil=lbm.D2Q9,
        thermal_stencil=lbm.D2Q5,
        buoyancy=jnp.array([0, buoyancy]),
    )

    sim = lbm.Simulation(domain, lattice, omegas)
    seed = 0
    key = jax.random.PRNGKey(seed)

    sim.set_initial_conditions(
        rho=jnp.ones((nx, ny, 1)),
        T=jax.random.uniform(key, (nx, ny, 1), minval=-0.05, maxval=0.05),
        u=jnp.zeros((nx, ny, 2)),
    )

    # Set the boundary conditions
    sim.set_boundary_conditions(
        lbm.BounceBackBoundary(
            "walls", sim.bottom | sim.top
        ),
        "FluidLattice",
    )

    sim.set_boundary_conditions(lbm.InletBoundary("bot", sim.bottom), "ThermalLattice")
    sim.set_boundary_conditions(lbm.InletBoundary("top", sim.top), "ThermalLattice")

    sim.update_boundary_condition("bot", {"m": 0.5}, "ThermalLattice")
    sim.update_boundary_condition("top", {"m": -0.5}, "ThermalLattice")

    init_ncfile(nc_path, sim)

    for i in trange(steps):
        sim.step()
        t = i * dt

        if i % io_frequency == 0:
            time_index = i // io_frequency
            write_ncfile(nc_path, time_index, t, sim.fluid_state)
            if jnp.isnan(sim.fluid_state.T).any():
                print("NaNs detected, stopping simulation.")
                break


