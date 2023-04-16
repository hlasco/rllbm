import jax.numpy as jnp
import jax

from rllbm import lbm as lbm
from rllbm.utils import MPLVideoRenderer


from tqdm.rich import trange
from tqdm import TqdmExperimentalWarning

from visualize import fig_constructor, fig_updater

import warnings
import wandb

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

class Diagnostic:
    def __init__(self, name, values):
        self.name = name
        self.min = jnp.min(values)
        self.max = jnp.max(values)
        self.mean = jnp.mean(values)
        self.std = jnp.std(values)
        
    def log(self, log_step):
        logpath = f"Diagnostics/{self.name}/"
        wandb.log({logpath + "min": self.min}, step=log_step)
        wandb.log({logpath + "max": self.max}, step=log_step)
        wandb.log({logpath + "mean": self.mean}, step=log_step)
        wandb.log({logpath + "std": self.std}, step=log_step)

def diagnostics(sim):
    ret = [
        Diagnostic("temperature", sim.fluid_state.T),
        Diagnostic("density", sim.fluid_state.rho),
        Diagnostic("velocity_x", sim.fluid_state.u[...,0]),
        Diagnostic("velocity_y", sim.fluid_state.u[...,1]),
    ]
    return ret

def main(n, pr, ra, buoy, log_wandb=False):


    if log_wandb:
        wandb.init(
            project="RLLBM-CFD",
            group="Rayleigh-Benard",
            config = {
                "N": n,
                "PR": pr,
                "RA": ra,
                "BOUY": buoy,
            }
        )

    nx = ny = n

    domain = lbm.Domain(shape=(nx, ny), bounds=(0.0, 1.0, 0.0, 1.0))

    dx = domain.dx
    dt = (buoy * dx) ** 0.5

    # Collision parameters
    viscosity = (pr / ra) ** 0.5 * dt / dx**2
    kappa = viscosity / pr

    convection_timescale = 1
    run_time = 100*convection_timescale
    # Number of steps to run
    steps = int(run_time / dt)
    # Frequency of writing to the netCDF file
    io_frequency = int(0.2*convection_timescale / dt)

    omegas = {
        "FluidLattice": 1.0 / (3 * viscosity + 0.5),
        "ThermalLattice": 1.0 / (3 * kappa + 0.5),
    }

    # Instantiate the lattice
    lattice = lbm.ThermalFluidLattice(
        fluid_stencil=lbm.D2Q9,
        thermal_stencil=lbm.D2Q5,
        buoyancy=jnp.array([0, buoy]),
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
    
    
    renderer = MPLVideoRenderer(
        fig_constructor=fig_constructor,
        fig_updater=fig_updater,
        live=False,
    )
    
    for i in trange(steps):
        sim.step()
        t = i * dt

        if i % io_frequency == 0:

            if jnp.isnan(sim.fluid_state.T).any():
                print("NaNs detected, stopping simulation.")
                break
            
            diags = diagnostics(sim)
            if log_wandb:
                log_step = i//io_frequency
                [diag.log(log_step) for diag in diags]
            renderer.render_frame(sim)
    
    fname = (
        "rb_NX-{}_PR-{:.1e}_RA-{:.1e}_BUOY-{:.1e}.mp4"
    ).format(nx, pr, ra, buoy)
    
    renderer.to_mp4(filename=fname, fps=15)
    if log_wandb:
        video = wandb.Video(fname)
        wandb.log({"video": video})
        wandb.finish()

if __name__ == "__main__":    
    for ra in [1e9]:
        main(n=96, pr=0.71, ra=ra, buoy=0.0001, log_wandb=False)


