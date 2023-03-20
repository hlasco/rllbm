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
def get_bottom_temperature(t, x, nx, freq_x0=0.05, freq_amp=0.02, amp=0.5, width=0.1):
    
    x_0 = 0.5*(x[-1] - x[0]) * jnp.sin(2 * jnp.pi * t * freq_x0)
    
    amp = amp * jnp.sin(x/nx * jnp.pi) * jnp.cos(2 * jnp.pi * t * freq_amp)
    
    return amp * jnp.exp(-0.5 * ((x - nx/2. - x_0) / (width*nx)) ** 2)

if __name__=="__main__":
    nx = 256
    ny = 256
    
    prandtl = 0.71
    rayleigh = 1e8
    buoyancy = 0.001
    
    dx = 1.0 / ( max(nx, ny) - 1.0)
    dt = (buoyancy * dx)**0.5
    
    viscosity = (prandtl / rayleigh) ** 0.5 * dt / dx ** 2
    kappa = 1.0 / (prandtl * rayleigh) ** 0.5 * dt / dx ** 2
    
    omegas = (
        1.0 / (3 * viscosity + 0.5),
        1.0 / (3 * kappa + 0.5),
    )
    
    collision_kwargs = {
        "gravity": jnp.array([buoyancy, 0]),
        "thermal_expansion": 1.0,
        "dt": dt,
    }
    
    sim = Simulation(nx, ny, dt, omegas, collision_kwargs)
    
    X, Y = jnp.meshgrid(sim.x, sim.y, indexing="ij")
    
    temperature = jnp.zeros((nx,ny))
    temperature = temperature.at[:,0].set(
        get_bottom_temperature(x=sim.x, t=0, nx=nx)
    )

    lattice = ConvectionLattice(
        NavierStokesLattice(D2Q9, (nx, ny)),
        AdvectionDiffusionLattice(D2Q5, (nx, ny)),
    )
    
    dfs = lattice.initialize(
        density=jnp.ones((nx, ny, 1)),
        velocity=jnp.zeros((nx, ny, 2)),
        temperature=temperature[...,jnp.newaxis],
    )
    
    sim.initialize(
        lattice,
        dfs,
        tracers = [jnp.array([0.5, 0.5])]
    )
    
    boundary_NSE = BoundaryDict(
        BounceBackBoundary(
            name="NSE No-Slip Walls",
            mask=(X == 0) | (X == nx-1) | (Y == 0) | (Y == ny-1)
        )
    )
    
    boundary_ADE = BoundaryDict(
        [
            BounceBackBoundary("ADE No-Slip Walls", (X == 0) | (X == nx-1)),
            DirichletBoundary("ADE Fixed Temperature Bottom Wall", (Y == 0)),
            DirichletBoundary("ADE Fixed Temperature Top Wall", (Y == ny-1)),
        ]
    )
    
    bc_kwargs = {
        "ADE Fixed Temperature Bottom Wall": {
            "fixed_value": get_bottom_temperature(x=sim.x, t=0, nx=nx),
        },
        "ADE Fixed Temperature Top Wall": {
            "fixed_value": jnp.zeros(nx),
        }
    }
    
    sim.set_boundary_conditions(
        (boundary_NSE, boundary_ADE),
        bc_kwargs
    )
    
    with netCDF4.Dataset("outputs_1.nc", "w", format='NETCDF4_CLASSIC') as f:
        f.createDimension('nx', sim.nx)
        f.createDimension('ny', sim.ny)
        f.createDimension('time', None)
        
        f.createVariable('t', 'f4', 'time')
        f.createVariable('x', 'f4', 'nx')[:] = sim.x
        f.createVariable('y', 'f4', 'ny')[:] = sim.y
        
        f.createVariable('temp', 'f4', ('time', 'nx', 'ny'))
        f.createVariable('ux', 'f4', ('time', 'nx', 'ny'))
        f.createVariable('uy', 'f4', ('time', 'nx', 'ny'))

    for i in trange(20000):
        sim.step()
        
        temperature_bot = get_bottom_temperature(x=sim.x, t=sim.time, nx=nx)
        sim.bc_kwargs["ADE Fixed Temperature Bottom Wall"]["fixed_value"] = temperature_bot
        if i%200 == 0:
            _, u, t = sim.get_macroscopics(sim.dfs)
            if jnp.isnan(t).any():
                break

            k = i//200
            with netCDF4.Dataset("outputs_1.nc", "a", format='NETCDF4_CLASSIC') as f:
                nctime = f.variables['t']
                nctime[k] = sim.time
                nctemp  = f.variables['temp']
                nctemp[k,:,:] = t
                ncuy = f.variables['ux']
                ncuy[k,:,:] = u[:,:,0]
                ncuy = f.variables['uy']
                ncuy[k,:,:] = u[:,:,1]
