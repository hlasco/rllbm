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


def write_ncfile(path, time_index, time, temperature):
    """Write the temperature data to a netCDF file.
    """
    with netCDF4.Dataset(path, "a") as ncfile:
        ncfile.variables["t"][time_index] = time
        ncfile.variables["temp"][time_index, :, :] = temperature[:, :, 0]

if __name__ == "__main__":
    nc_path = "outputs.nc"
    
    nx, ny = 256, 256

    domain = lbm.Domain(
        shape=(nx, ny),
        bounds=(0., 1., 0., 1.)
    )

    dx = domain.dx
    dt = dx ** 0.5

    prandtl = 0.71
    rayleigh = 1e11
    thermal_expansion = 0.005
    gravity = 9.81
    buoyancy = gravity * thermal_expansion

    # Collision parameters
    viscosity = (buoyancy * prandtl / rayleigh)**0.5 * dt / dx**2
    kappa = viscosity / prandtl

    convection_timescale = 1.0 / buoyancy
    run_time = 100 * convection_timescale
    # Number of steps to run
    steps = int(run_time / dt)
    # Frequency of writing to the netCDF file
    io_frequency = int(convection_timescale / dt)

    omegas = (
        1.0 / (3 * viscosity + 0.5),
        1.0 / (3 * kappa + 0.5),
    )

    # ConvectionLattice needs to know those parameters in order to calculate the
    # collision terms
    collision_kwargs = {
        "timestep": dt,
        "gravity": jnp.array([0, gravity]),
        "thermal_expansion": thermal_expansion,
    }

    # Instantiate the simulation
    sim = lbm.Simulation(domain, omegas, collision_kwargs)

    # Instantiate the lattice
    lattice = lbm.ThermalFluidLattice(
        fluid_stencil=lbm.D2Q9,
        thermal_stencil=lbm.D2Q5,
    )

    # Initialize the density functions
    dfs = lattice.initialize(
        rho=jnp.ones((nx, ny, 1)),
        T=jnp.zeros((nx, ny, 1)),
        u=jnp.zeros((nx, ny, 2)),
    )
    sim.initialize(lattice, dfs)

    # Set the boundary conditions
    fluid_bc = lbm.BoundaryDict(
        lbm.BounceBackBoundary("No-Slip Walls", sim.left | sim.right | sim.bottom | sim.top)
    )
    fluid_bc_kwargs = {}
    
    thermal_bc = lbm.BoundaryDict(
        [
            lbm.BounceBackBoundary("No-Slip Walls", sim.bottom | sim.top),
            lbm.InletBoundary("Left Wall", sim.left),
            lbm.InletBoundary("Right Wall", sim.right),
        ]
    )
    
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
        t = i * dt
        # Update the left wall temperature
        sim.bc_kwargs[1]["Left Wall"]["m"] = get_wall_temperature(t / convection_timescale, sim.y)

        if i % io_frequency == 0:
            fluid_state = sim.get_macroscopics(sim.dfs)
            
            if jnp.isnan(fluid_state.T).any():
                print("NaNs detected, stopping simulation.")
                break

            time_index = i // io_frequency
            write_ncfile(nc_path, time_index, t, fluid_state.T)
