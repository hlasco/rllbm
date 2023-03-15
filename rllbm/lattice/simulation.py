import jax.numpy as jnp
import jax
from rllbm.lattice import D2Q5, D2Q9, stream, collide_NSE_TDE, no_slip_bc
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
        self.no_slip_NSE = (X == 0) | (X == self.nx-1) | (Y == 0) | (Y == self.ny-1)
        self.no_slip_TDE = (X == 0) | (X == self.nx-1)

        self.prandtl = prandtl
        self.rayleigh = rayleigh
        self.buyancy = jnp.array([buyancy, 0])
        
        self.dx = 1.0 / ( max(nx, ny) - 1.0)
        self.dt = (buyancy * self.dx)**0.5
        
        viscosity = (prandtl / rayleigh) ** 0.5 * self.dt / self.dx ** 2
        kappa = 1.0 / (prandtl * rayleigh) ** 0.5 * self.dt / self.dx ** 2

        self.omega_NSE = 1.0 / (3 * viscosity + 0.5)
        self.omega_TDE = 1.0 / (3 * kappa + 0.5)
        
        self.d2q5 = D2Q5()
        self.d2q9 = D2Q9()
        
        self.time = 0

    
    def initialize(self, temperature, tracers=None):
        self.dist_function_NSE = jnp.ones((self.nx, self.ny, self.d2q9.size))
        self.dist_function_NSE *= self.d2q9.weights[jnp.newaxis, jnp.newaxis, :]
        
        self.dist_function_TDE = temperature[:,:,jnp.newaxis] * jnp.ones((self.nx, self.ny, self.d2q5.size))
        self.dist_function_TDE *= self.d2q5.weights[jnp.newaxis, jnp.newaxis, :]
        
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
    
    def get_macroscopics(self):
        return self._get_microscopics(
            self.dist_function_NSE,
            self.dist_function_TDE,
        )
    
    @partial(jax.jit, static_argnums=(0))
    def _get_microscopics(self, dist_function_NSE, dist_function_TDE):
        density = self.d2q9.get_moment(dist_function_NSE, 0)
        velocity = self.d2q9.get_moment(dist_function_NSE, 1) / density[..., jnp.newaxis]
        temperature = self.d2q9.get_moment(dist_function_TDE, 0)
        return density, velocity, temperature

    @partial(jax.jit, static_argnums=(0))
    def _step(
        self,
        dist_function_NSE,
        dist_function_TDE,
        temperature_top,
        temperature_bot,
    ):
        df_NSE_new = dist_function_NSE
        df_TDE_new = dist_function_TDE

        df_NSE_new, df_TDE_new = collide_NSE_TDE(
            df_NSE_new, self.d2q9, self.omega_NSE,
            df_TDE_new, self.d2q5, self.omega_TDE, self.buyancy,
        )

        df_NSE_new = stream(df_NSE_new, self.d2q9)
        df_TDE_new = stream(df_TDE_new, self.d2q5)

        df_NSE_new = no_slip_bc(df_NSE_new, dist_function_NSE, self.no_slip_NSE, self.d2q9)
        df_TDE_new = no_slip_bc(df_TDE_new, dist_function_TDE, self.no_slip_TDE, self.d2q5)

        df_TDE_new = df_TDE_new.at[:,-1,4].set(
            temperature_top
            - df_TDE_new[:,-1,0]
            - df_TDE_new[:,-1,1]
            - df_TDE_new[:,-1,2]
            - df_TDE_new[:,-1,3]
        )

        df_TDE_new = df_TDE_new.at[:,0,2].set(
            temperature_bot
            - df_TDE_new[:,0,0]
            - df_TDE_new[:,0,1]
            - df_TDE_new[:,0,3]
            - df_TDE_new[:,0,4]
        )

        return df_NSE_new, df_TDE_new
    
    @partial(jax.jit, static_argnums=(0))
    def _update_tracers(
        self,
        tracers,
        dist_function_NSE,
    ):
        density = self.d2q9.get_moment(dist_function_NSE, 0)
        velocity = self.d2q9.get_moment(dist_function_NSE, 1) / density[..., jnp.newaxis]
        for tracer_id, tracer in enumerate(tracers):
            idx, idy = jnp.floor(tracer[0] / self.dx), jnp.floor(tracer[1] / self.dx)
            tracer += velocity[idx.astype(int)%self.nx, idy.astype(int)%self.ny, :] * self.dt
            tracer = tracer.at[0].set(tracer[0] % 1.0)
            tracer = tracer.at[1].set(tracer[1] % 1.0)
            tracers[tracer_id] = tracer
            
        return tracers