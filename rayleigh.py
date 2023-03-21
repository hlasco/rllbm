import jax.numpy as jnp
import jax

from rllbm.lattice import (
    Simulation,
    NavierStokesLattice,
    AdvectionDiffusionLattice,
    ConvectionLattice,
    D2Q9,
    D2Q5,
    BoundaryDict,
    DirichletBoundary,
    BounceBackBoundary,
)
from tqdm.rich import trange
from tqdm import TqdmExperimentalWarning
import netCDF4

import warnings

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)


@jax.jit
def get_bottom_temperature(t, x, nx, freq_x0=0.005, freq_amp=0.02, amp=0.5, width=0.1):
    x_0 = 0.5 * (x[-1] - x[0]) * jnp.sin(2 * jnp.pi * t * freq_x0)
    amp = amp * jnp.sin(x / nx * jnp.pi) * jnp.cos(2 * jnp.pi * t * freq_amp)
    return amp * jnp.exp(-0.5 * ((x - nx / 2.0 - x_0) / (width * nx)) ** 2)

@jax.jit
def update_tracers(tracers, velocity, dt, dx, nx, ny):
   for tracer_id, tracer in enumerate(tracers):
       idx, idy = jnp.floor(tracer[0] / dx), jnp.floor(tracer[1] / dx)
       tracer += velocity[idx.astype(int)%nx, idy.astype(int)%ny, :] * dt
       tracer = tracer.at[0].set(tracer[0] % 1.0)
       tracer = tracer.at[1].set(tracer[1] % 1.0)
       tracers[tracer_id] = tracer
   return tracers


def init_ncfile(path, sim, tracers):
    with netCDF4.Dataset(path, "w", format="NETCDF4") as ncfile:
        ncfile.createDimension("nx", sim.nx)
        ncfile.createDimension("ny", sim.ny)
        ncfile.createDimension("time", None)
        ncfile.createDimension("tracer_pos", 2)
        ncfile.createDimension("tracer_id", len(tracers))

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
        ncfile.createVariable(
            "curl",
            "f4",
            ("time", "nx", "ny"),
            compression="zlib",
            least_significant_digit=6,
            fill_value=0,
        ),
        ncfile.createVariable(
            "ux",
            "f4",
            ("time", "nx", "ny"),
            compression="zlib",
            least_significant_digit=3,
        )
        ncfile.createVariable(
            "uy",
            "f4",
            ("time", "nx", "ny"),
            compression="zlib",
            least_significant_digit=3,
        )

        ncfile.createVariable(
            "tracers",
            "f4",
            ("tracer_id", "time", "tracer_pos"),
        )

        ncfile.variables["x"][:] = sim.x / sim.nx
        ncfile.variables["y"][:] = sim.y / sim.ny


def write_ncfile(path, time_index, time, velocity, temperature, curl, tracers):
    with netCDF4.Dataset(path, "a") as ncfile:
        ncfile.variables["t"][time_index] = time
        ncfile.variables["temp"][time_index, :, :] = temperature[:, :, 0]
        ncfile.variables["curl"][time_index, :, :] = curl
        ncfile.variables["ux"][time_index, :, :] = velocity[:, :, 0]
        ncfile.variables["uy"][time_index, :, :] = velocity[:, :, 1]
        
        for tracer_id, tracer in enumerate(tracers):
            ncfile.variables["tracers"][tracer_id, time_index, :] = tracer


if __name__ == "__main__":
    nc_path = "outputs_1.nc"

    nx = 64
    ny = 64

    prandtl = 0.71
    rayleigh = 1e8
    buoyancy = 0.001

    dx = 1.0 / (max(nx, ny) - 1.0)
    dt = (buoyancy * dx) ** 0.5
    
    simulation_time = 300.0
    simulation_steps = int(simulation_time / dt)
    simulation_io_frequency = int(1.0 / dt)

    viscosity = (prandtl / rayleigh) ** 0.5 * dt / dx**2
    kappa = 1.0 / (prandtl * rayleigh) ** 0.5 * dt / dx**2

    omegas = (
        1.0 / (3 * viscosity + 0.5),
        1.0 / (3 * kappa + 0.5),
    )

    collision_kwargs = {
        "gravity": jnp.array([0, buoyancy]),
        "thermal_expansion": 1.0,
        "dt": dt,
    }

    sim = Simulation(nx, ny, dt, omegas, collision_kwargs)

    X, Y = jnp.meshgrid(sim.x, sim.y, indexing="ij")

    lattice = ConvectionLattice(
        NavierStokesLattice(D2Q9, (nx, ny)),
        AdvectionDiffusionLattice(D2Q5, (nx, ny)),
    )

    dfs = lattice.initialize(
        density=jnp.ones((nx, ny, 1)),
        velocity=jnp.zeros((nx, ny, 2)),
        temperature=jnp.zeros((nx, ny, 1)),
    )

    sim.initialize(lattice, dfs)

    boundary_NSE = BoundaryDict(
        [
            BounceBackBoundary(
                "NSE No-Slip Walls", (X == 0) | (X == nx - 1) | (Y == 0) | (Y == ny - 1)
            ),
        ]
    )

    boundary_ADE = BoundaryDict(
        [
            BounceBackBoundary("ADE No-Slip Walls", (Y == 0) | (Y == ny - 1)),
            DirichletBoundary("ADE Fixed Temperature Bottom Wall", (X == 0)),
            DirichletBoundary("ADE Fixed Temperature Top Wall", (X == nx - 1)),
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

    tracers=[
        jnp.array([0.4, 0.5]),
        jnp.array([0.5, 0.5]),
        jnp.array([0.6, 0.5]),
        jnp.array([0.5, 0.4]),
        jnp.array([0.5, 0.6]),
    ]
    
    init_ncfile(nc_path, sim, tracers)

    for i in trange(simulation_steps):
        sim.step()
        
        _, velocity, temperature = sim.get_macroscopics(sim.dfs)
        
        tracers = update_tracers(tracers, velocity, dt, dx, nx, ny)

        sim.bc_kwargs["ADE Fixed Temperature Bottom Wall"][
            "fixed_value"
        ] = get_bottom_temperature(x=sim.y, t=sim.time, nx=ny)

        if i % simulation_io_frequency == 0:
            
            if jnp.isnan(temperature).any():
                break

            _, dudy = jnp.gradient(velocity[:, :, 0], dx)
            dvdx, _ = jnp.gradient(velocity[:, :, 1], dx)
            curl = dudy - dvdx

            time_index = i // simulation_io_frequency
            write_ncfile(nc_path, time_index, sim.time, velocity, temperature, curl, tracers)
