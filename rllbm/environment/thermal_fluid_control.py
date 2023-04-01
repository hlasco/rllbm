
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from rllbm import lbm
from rllbm.environment import LBMEnv

import holoviews as hv

from PIL import Image
import io

__all__ = ["ThermalFluidControl"]

renderer = hv.renderer('matplotlib')

@jax.jit
def get_wall_temperature(x, bc_params):
    """
    """
    x_peak, amp, width = bc_params[0], bc_params[1], bc_params[2]
    x_min = x[0]
    x_max = x[-1]
    
    x_ = (x - x_min) / (x_max - x_min)

    ret = amp * jnp.exp(-0.5 * ((x_ - x_peak) / (width)) ** 2) * jnp.sin(x_  * jnp.pi)
    return ret

DEFAULT_PARAMS = {
    "nx": 128,
    "ny": 128,
    "temperature_amplitude": 1.0,
    "run_time": 50.0,
    "step_time": 1.0,
    "prandtl": 0.71,
    "rayleigh": 1e10,
    "thermal_expansion": 0.005,
    "gravity": 9.81,
    "sensor_shape": (4,4),
    "stream_sensors": False,
}

class ThermalFluidControl(LBMEnv):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 5}

    def __init__(
        self,
        params=DEFAULT_PARAMS,
        render_mode=None
    ):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        super().__init__(params)
        
        convection_timescale = 1.0 / (self.gravity * self.thermal_expansion)
        
        run_time = self.run_time * convection_timescale
        step_time = self.step_time * convection_timescale
        
        dt = self.sim.dx ** 0.5

        self.sim_steps = int(run_time / dt)
        self.sim_steps_per_env_step = int(step_time / dt)
        
        self.bc_params_bounds = {
            "min": jnp.array([0.0, -0.5*self.temperature_amplitude, 0.05]),
            "max": jnp.array([1.0,  0.5*self.temperature_amplitude, 0.1]),
        }

        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        self.action_space = gym.spaces.Box(low = -1.0, high = 1.0, shape = (3,))
        
    def initialize_sim(self) -> lbm.Simulation:
        """Initialize the simulation.
        """

        # Define the domain
        domain = lbm.Domain(
            shape=(self.nx, self.ny),
            bounds=(0., 1.0, 0., 1.0)
        )
        
        # Define relaxation parameters
        dx = domain.dx
        dt = dx ** 0.5
        
        buoyancy = self.gravity * self.thermal_expansion
        
        viscosity = (buoyancy * self.prandtl / self.rayleigh)**0.5 * dt / dx**2
        kappa = viscosity / self.prandtl
        omegas = (
            1.0 / (3 * viscosity + 0.5),
            1.0 / (3 * kappa + 0.5),
        )

        # Define the collision parameters
        collision_kwargs = {
            "timestep": dt,
            "gravity": jnp.array([0, self.gravity]),
            "thermal_expansion": self.thermal_expansion,
        }

        sim = lbm.Simulation(domain, omegas, collision_kwargs)
        
        lattice = lbm.ThermalFluidLattice(
            fluid_stencil=lbm.D2Q9,
            thermal_stencil=lbm.D2Q5,
        )

        # Initialize the density functions
        initial_condition = lattice.initialize(
            rho=jnp.ones((self.nx, self.ny, 1)),
            T=jnp.zeros((self.nx, self.ny, 1)),
            u=jnp.zeros((self.nx, self.ny, 2)),
        )
        
        sim.initialize(lattice, initial_condition)
        
        # Set the boundary conditions
        fluid_bc = lbm.BoundaryDict()
        fluid_bc.add(
            lbm.BounceBackBoundary(
                "No-Slip Walls", sim.bottom | sim.top | sim.left | sim.right
            ), 
        )

        thermal_bc = lbm.BoundaryDict()
        thermal_bc.add(
            [
                lbm.InletBoundary("Top Wall", sim.top),
                lbm.InletBoundary("Bottom Wall", sim.bottom),
                lbm.InletBoundary("Left Wall", sim.left),
                lbm.InletBoundary("Right Wall", sim.right),
            ]
        )
        fluid_bc_kwargs = {}

        thermal_bc_kwargs = {
            "Left Wall": {"m": 0.0},
            "Right Wall": {"m": 0.0},
            "Bottom Wall": {"m": 0.0},
            "Top Wall": {"m": 0.0},
        }
        
        sim.set_boundary_conditions(
            (fluid_bc, thermal_bc),
            (fluid_bc_kwargs, thermal_bc_kwargs),
        )
        
        return sim


    def reset(self, seed=None, options=None):
        initial_condition = self.sim.lattice.initialize(
            rho=jnp.ones((self.nx, self.ny, 1)),
            T=jnp.zeros((self.nx, self.ny, 1)),
            u=jnp.zeros((self.nx, self.ny, 2)),
        )

        super().reset(seed=seed, initial_condition=initial_condition)
        
        self.bc_params = jax.random.uniform(
            self.random_key,
            shape=(3,),
            minval=self.bc_params_bounds["min"],
            maxval=self.bc_params_bounds["max"],
        )
        
        self.add_grid_tracers("sensors", self.sensor_shape, stream=self.stream_sensors)
        self.add_random_tracers("current", 1, stream=True)
        self.add_random_tracers("target", 1, stream=True)

        info = None
        return 0.0, info

    def step(self, action):
        self.bc_params += (self.bc_params_bounds["min"] - self.bc_params_bounds["max"]) / 15 * action
        self.bc_params = np.clip(self.bc_params, self.bc_params_bounds["min"], self.bc_params_bounds["max"])
        
        wall_tenperature = get_wall_temperature(self.sim.y, self.bc_params)
        self.sim.bc_kwargs[1]["Left Wall"]["m"] = wall_tenperature

        for _ in range(self.sim_steps_per_env_step):
            self.sim.step()
            self.sim_step += 1
            self.fluid_state = self.sim.get_macroscopics(self.sim.dfs)
            self.tracers = lbm.update_tracers(self.sim, self.fluid_state, self.tracers)

        terminated = self.sim_step > self.sim_steps

        if self.simulation_failed():
            print("Simulation crashed")
            terminated = True

        observation = 0.0
        reward = 1 if terminated else 0
        info = None
        return observation, reward, terminated, False, info

    def render(self):
        if not hasattr(self, "fluid_state"):
            self.fluid_state = self.sim.get_macroscopics(self.sim.dfs)
        if self.render_mode == "rgb_array":
            frame = self._render_frame()
            return frame

    def _render_frame(self):
        x = self.sim.x
        y = self.sim.y 
        
        cmax_img = 0.5 * self.gravity * self.thermal_expansion * self.temperature_amplitude
        cmax_crv = 0.5 * self.temperature_amplitude
        
        img = hv.Image((x, y, self.fluid_state.T[:,:,0].T))
        img = img.opts(
            cmap="RdBu_r",
            clim=(-cmax_img, cmax_img),
            ylabel="",
            yticks=0,
            colorbar=True,
            clabel="Fluid Temperature",
            cbar_extend  = "neither",
            aspect=1.0,
        )
        curve = hv.Curve((self.fluid_state.T[0,:], y), "T_0", "y",).opts(
            aspect = 1.0/4,
            xlim=(-cmax_crv, cmax_crv),
            xticks=(cmax_crv, 0, cmax_crv),
            xlabel="Wall Temperature",
        )

        sensors = self.tracers.get("sensors")
        sensors = jnp.stack([t.x for t in sensors], axis=0)
        sensors = hv.Scatter(sensors, label='Sensors').opts(
            color='#566573',
            marker='D',
            alpha=0.5,
        )

        current = self.tracers.get("current")
        current = jnp.stack([t.x for t in current], axis=0)
        current = hv.Scatter(current, label='Current').opts(
            color='#7C5104',
            marker='o',
            alpha=0.8,
        )
        target = self.tracers.get("target")
        target = jnp.stack([t.x for t in target], axis=0)
        target = hv.Scatter(target, label='Target').opts(
            color='#2ECC71',
            marker='o',
            alpha=0.8,
        )

        overlay = (img * current * target * sensors).opts(
            legend_cols=3,
            legend_position='top',
            fontsize={"labels": 10},
        )

        layout = hv.Layout(curve + overlay).cols(2)
        layout = layout.opts(
            vspace = 0,
            hspace = 0,
            tight=True,
            aspect_weight=True,
            sublabel_format="",
            fig_size=150,
        )

        png, _ = renderer(layout, fmt='png', dpi=120)
        image = Image.open(io.BytesIO(png))
        return np.array(image)

