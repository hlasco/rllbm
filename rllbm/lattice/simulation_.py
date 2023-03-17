import jax.numpy as jnp
import jax
from rllbm.lattice import BounceBackBoundary, AntiBounceBackBoundary
from rllbm.lattice import stream, collide_NSE_TDE, no_slip_bc
from rllbm.lattice import D2Q5_, D2Q9_
from rllbm.lattice import NavierStokesLattice, AdvectionDiffusionLattice
from functools import partial

class RayleighBenardSimulation:
    def __init__(
        self,
        nx: int,
        ny: int,
        prandtl: float,
        rayleigh: float,
        buyancy: float,
    ) -> None:
        self.nx = nx
        self.ny = ny

        self.x = jnp.arange(self.nx)
        self.y = jnp.arange(self.ny)
        X, Y = jnp.meshgrid(self.x, self.y, indexing="ij")
        
        
        no_slip_NSE = (X == 0) | (X == self.nx-1) | (Y == 0) | (Y == self.ny-1)
        self.no_slip_NSE = BounceBackBoundary(no_slip_NSE)
        self.no_collision_NSE = self.no_slip_NSE.no_collision_mask
        self.no_streaming_NSE = self.no_slip_NSE.no_streaming_mask
        
        no_slip_TDE = (X == 0) | (X == self.nx-1)
        self.no_slip_TDE = BounceBackBoundary(no_slip_TDE)
        dirichlet_TDE = (Y == 0) | (Y == self.ny-1)
        self.dirichlet_TDE = AntiBounceBackBoundary(dirichlet_TDE)
        self.no_collision_TDE = self.no_slip_TDE.no_collision_mask * self.dirichlet_TDE.no_collision_mask
        self.no_streaming_TDE = self.no_slip_TDE.no_streaming_mask * self.dirichlet_TDE.no_streaming_mask

        self.prandtl = prandtl
        self.rayleigh = rayleigh
        self.buyancy = jnp.array([buyancy, 0])
        
        self.dx = 1.0 / ( max(nx, ny) - 1.0)
        self.dt = (buyancy * self.dx)**0.5
        
        viscosity = (prandtl / rayleigh) ** 0.5 * self.dt / self.dx ** 2
        kappa = 1.0 / (prandtl * rayleigh) ** 0.5 * self.dt / self.dx ** 2

        self.omega_NSE = 1.0 / (3 * viscosity + 0.5)
        self.omega_TDE = 1.0 / (3 * kappa + 0.5)
        
        self.time = 0

    
    def initialize(self, temperature, tracers=None):
        self.nse = NavierStokesLattice(D2Q9_, (self.nx, self.ny))
        self.nse.initialize(
            density=jnp.ones((self.nx, self.ny)),
            velocity=jnp.zeros((self.nx, self.ny)),
        )
        
        self.ade = AdvectionDiffusionLattice(D2Q5_, (self.nx, self.ny))
        self.ade.initialize(
            temperature=temperature,
            velocity=jnp.zeros((self.nx, self.ny)),
        )
        
        self.temperature_bot = temperature[:, 0]
        self.temperature_top = temperature[:,-1]
        
        self.tracers = tracers
        
        self.time = 0
    
    def step(self):
        self.dist_function_NSE, self.dist_function_TDE = self._step(
            self.dist_function_NSE,
            self.dist_function_TDE,
            self.temperature_top,
            self.temperature_bot,
        )
        
        if self.tracers:
            self.tracers = self._update_tracers(self.tracers, self.dist_function_NSE)
        
        self.time += self.dt
    
    @partial(jax.jit, static_argnums=(0))
    def get_microscopics(self):
        density = self.nse.get_moment(0)
        velocity = self.nse.get_moment(1) / density[..., jnp.newaxis]
        temperature = self.ade.get_moment(0)
        return density, velocity, temperature

    @partial(jax.jit, static_argnums=(0))
    def _step(
        self,
        nse,
        ade,
        temperature_top,
        temperature_bot,
    ):
        nse.df = nse.df.at[...].set(
            jnp.where(
                self.no_collision_NSE,
                nse.df,
                
            )
        )
        df_NSE_pc, df_TDE_pc = collide_NSE_TDE(
            dist_function_NSE, D2Q9_, self.omega_NSE,
            dist_function_TDE, D2Q5_, self.omega_TDE, self.buyancy,
        )

        df_NSE_new = stream(df_NSE_pc, D2Q9_)
        df_TDE_new = stream(df_TDE_pc, D2Q5_)

        df_NSE_new = no_slip_bc(df_NSE_new, dist_function_NSE, self.no_slip_NSE, D2Q9_)
        df_TDE_new = no_slip_bc(df_TDE_new, dist_function_TDE, self.no_slip_TDE, D2Q5_)

        for i in range(5):
            df_TDE_new = df_TDE_new.at[:,-1,i].set(
                -df_TDE_pc[:,-1,D2Q5_.opposite_indices[i]] + 1.0/5 * temperature_top
            )
            df_TDE_new = df_TDE_new.at[:,0,i].set(
                -df_TDE_pc[:,-1,D2Q5_.opposite_indices[i]] + 1.0/5 * temperature_bot
            )

        return df_NSE_new, df_TDE_new
    
    @partial(jax.jit, static_argnums=(0))
    def _update_tracers(
        self,
        tracers,
        dist_function_NSE,
    ):
        density = D2Q9_.get_moment(dist_function_NSE, 0)
        velocity = D2Q9_.get_moment(dist_function_NSE, 1) / density[..., jnp.newaxis]
        for tracer_id, tracer in enumerate(tracers):
            idx, idy = jnp.floor(tracer[0] / self.dx), jnp.floor(tracer[1] / self.dx)
            tracer += velocity[idx.astype(int)%self.nx, idy.astype(int)%self.ny, :] * self.dt
            tracer = tracer.at[0].set(tracer[0] % 1.0)
            tracer = tracer.at[1].set(tracer[1] % 1.0)
            tracers[tracer_id] = tracer
            
        return tracers