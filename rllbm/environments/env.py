
import abc
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from rllbm import lbm

import holoviews as hv

import time

from PIL import Image
import io
from functools import partial

renderer = hv.renderer('matplotlib')

@jax.jit
def get_wall_temperature(x, x_peak, amp, width):
    """
    Function to calculate the bottom temperature at the current time step (t) and spatial location (y)
        nx is the number of grid points in the x-direction
        freq_x0 is the frequency of the sinusoidal variation in the x-direction
        freq_amp is the frequency of the sinusoidal variation in the amplitude
        amp is the maximum amplitude of the temperature
        width is the width of the temperature profile
    """
    x_min = x[0]
    x_max = x[-1]
    
    x_ = (x - x_min) / (x_max - x_min)

    ret = amp * jnp.exp(-0.5 * ((x_ - x_peak) / (width)) ** 2) * jnp.sin(x_  * jnp.pi)
    return ret

@partial(jax.jit, static_argnums=(0))
def update_tracers(sim, dfs, tracers):
    fluid_state = sim.get_macroscopics(dfs)
    for tracer_type in tracers.keys():
        for i, tracer in enumerate(tracers.get(tracer_type)):
            idx = jnp.floor(tracer.x / sim.dx).astype(int)
            u = fluid_state.u[idx[0], idx[1]]
            
            tracer.x += u * sim.dx * tracer.stream
            tracer.x = jnp.clip(
                tracer.x,
                a_min = jnp.array([sim.dx, sim.dx]),
                a_max = jnp.array([1.0-sim.dx, 1.0-sim.dx])
            )
            tracer.obs = jnp.concatenate(
                [state[idx[0], idx[1]] for state in fluid_state]
            )

            tracers.set_tracer(tracer_type, i, tracer)

    return tracers

class LBMEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Simulation parameters
        nx, ny = 256, 256
        domain = lbm.Domain(
            shape=(nx, ny),
            bounds=(0., 1.0, 0., 1.0)
        )

        dx = domain.dx
        dt = dx ** 0.5

        prandtl = 0.71
        rayleigh = 1e11
        thermal_expansion = 0.005
        gravity = 9.81
        buoyancy = gravity * thermal_expansion
        
        convection_timescale = 1.0 / buoyancy
        run_time = 30 * convection_timescale
        # Number of steps to run
        self.sim_steps = int(run_time / dt)
        self.sim_steps_per_env_step = int(convection_timescale / dt)

        # Collision parameters
        viscosity = (buoyancy * prandtl / rayleigh)**0.5 * dt/ dx**2
        kappa = viscosity / prandtl
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
        self.sim = lbm.Simulation(domain, omegas, collision_kwargs)

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(1,))

        self.bc_params_bounds = {
            "low": np.array([0.0, -0.5, 0.05]),
            "high": np.array([1.0, 0.5, 0.1]),
        }
        self.bc_params = np.random.uniform(
            low=self.bc_params_bounds["low"],
            high=self.bc_params_bounds["high"],
        )

        self.action_space = gym.spaces.Box(
            low = -1.0,
            high = 1.0,
            shape = (3,),
        )

        print("Initialized environment")


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Instantiate the lattice
        self.lattice = lbm.ThermalFluidLattice(
            fluid_stencil=lbm.D2Q9,
            thermal_stencil=lbm.D2Q5,
        )

        # Initialize the density functions
        dfs = self.lattice.initialize(
            rho=jnp.ones((*self.sim.shape, 1)),
            T=jnp.zeros((*self.sim.shape, 1)),
            u=jnp.zeros((*self.sim.shape, 2)),
        )
        
        self.sim.initialize(self.lattice, dfs)

        # Set the boundary conditions
        fluid_bc = lbm.BoundaryDict(
            [
                lbm.BounceBackBoundary(
                    "No-Slip Walls", self.sim.bottom | self.sim.top | self.sim.left | self.sim.right
                ),
            ]
        )
        thermal_bc = lbm.BoundaryDict(
            [
                lbm.BounceBackBoundary("No-Slip Walls", self.sim.bottom | self.sim.top),
                lbm.InletBoundary("Left Wall", self.sim.left),
                lbm.InletBoundary("Right Wall", self.sim.right),
            ]
        )
        fluid_bc_kwargs = {}

        thermal_bc_kwargs = {
            "Left Wall": {"m": get_wall_temperature(self.sim.y, *self.bc_params)},
            "Right Wall": {"m": 0.0},
        }
        
        self.sim.set_boundary_conditions(
            (fluid_bc, thermal_bc),
            (fluid_bc_kwargs, thermal_bc_kwargs),
        )

        self.sim_step = 0

        self.tracers = lbm.TracerCollection()

        for _ in range(50):
            pos = np.random.uniform(0.05, 0.95, size=(2,))
            pos = jnp.array(pos)
            self.tracers.add("sensors", lbm.Tracer(x=pos, stream=True))

        pos = np.random.uniform(0.05, 0.95, size=(2,))
        self.tracers.add("current", lbm.Tracer(x = jnp.asarray(pos), stream=True))

        pos = np.random.uniform(0.05, 0.95, size=(2,))
        self.tracers.add("target", lbm.Tracer(x = jnp.asarray(pos), stream=True))

        info = None
        return 0.0, info

    def step(self, action):
        self.bc_params += (self.bc_params_bounds["high"] - self.bc_params_bounds["low"]) / 15 * action
        self.bc_params = np.clip(self.bc_params, self.bc_params_bounds["low"], self.bc_params_bounds["high"])
        wall_tenperature = get_wall_temperature(self.sim.y, self.bc_params[0], self.bc_params[1], self.bc_params[2])
        self.sim.bc_kwargs[1]["Left Wall"]["m"] = wall_tenperature

        for _ in range(self.sim_steps_per_env_step):
            self.sim.step()
            self.sim_step += 1

            self.tracers = update_tracers(self.sim, self.sim.dfs, self.tracers)

        terminated = self.sim_step > self.sim_steps
        current = self.tracers.get("current")
        print(current[0].obs)
        if jnp.isnan(current[0].obs).any():
            print("Simulation crashed")
            terminated = True

        observation = 0.0
        reward = 1 if terminated else 0
        info = None
        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            frame = self._render_frame()
            return frame

    def _render_frame(self):
        fluid_state = self.sim.get_macroscopics(self.sim.dfs)
        x = self.sim.x
        y = self.sim.y 
        img = hv.Image((x, y, fluid_state.T[:,:,0].T))
        img = img.opts(
            cmap="RdBu_r",
            clim=(-0.05, 0.05),
            ylabel="",
            yticks=0,
            colorbar=True,
            clabel="Fluid Temperature",
            cbar_extend  = "neither",
            aspect=1.0,
        )
        curve = hv.Curve((fluid_state.T[0,:], y), "T_0", "y",).opts(
            aspect = 1.0/4,
            xlim=(-0.55, 0.55),
            xticks=(-0.5, 0, 0.5),
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

