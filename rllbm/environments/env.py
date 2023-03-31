
import abc
import gymnasium as gym
import time
import jax
import jax.numpy as jnp
import numpy as np

from typing import Callable

from rllbm import lbm

import holoviews as hv

from holoviews.operation.datashader import rasterize

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
    density, velocity, temperature = sim.get_macroscopics(dfs)
    for tracer_type in tracers.keys():
        for i, tracer in enumerate(tracers.get(tracer_type)):
            idx = jnp.floor(tracer.x / sim.dx).astype(int)
            d = density[idx[0], idx[1],0]
            t = temperature[idx[0], idx[1],0]
            v = velocity[idx[0], idx[1],:]
            
            tracer.x += v * sim.dx * tracer.stream
            tracer.x = jnp.clip(
                tracer.x,
                a_min = jnp.array([sim.dx, sim.dx]),
                a_max = jnp.array([1.0-sim.dx, 1.0-sim.dx])
            )
            tracer.obs = jnp.stack([d, t, v[0], v[1]], axis=-1)

            tracers.set_tracer(tracer_type, i, tracer)
            print(tracer.x)

    return tracers

    for i in range(len(tracers)):
        idx = jnp.floor(tracers[i].x / sim.dx).astype(int)
        tracers[i].x += velocity[idx[0], idx[1], :] * sim.dx * tracers[i].stream
        tracers[i].x = jnp.clip(
            tracers[i].x,
            a_min = jnp.array([sim.dx, sim.dx]),
            a_max = jnp.array([1.0-sim.dx, 1.0-sim.dx])
        )
        tracers[i].obs = jnp.stack([density[idx[0], idx[1],0], temperature[idx[0], idx[1],0], velocity[idx[0], idx[1],0], velocity[idx[0], idx[1],1]], axis=-1)
    return tracers

class LBMEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Simulation parameters
        nx = 256
        ny = 256

        dx = 1.0 / (max(nx, ny)-1)
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
        self.sim = lbm.Simulation(nx, ny, dt, omegas, collision_kwargs)

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
        X, Y = jnp.meshgrid(self.sim.x, self.sim.y, indexing="ij")

        # Instantiate the lattice
        self.lattice = lbm.ConvectionLattice(
            fluid_stencil=lbm.D2Q9,
            thermal_stencil=lbm.D2Q5,
            shape=(self.sim.nx, self.sim.ny),
        )

        # Initialize the density functions
        dfs = self.lattice.initialize(
            density=jnp.ones((self.sim.nx, self.sim.ny, 1)),
            velocity=jnp.zeros((self.sim.nx, self.sim.ny, 2)),
            temperature=jnp.zeros((self.sim.nx, self.sim.ny, 1)),
        )
        
        self.sim.initialize(self.lattice, dfs)

        # Set the boundary conditions
        fluid_bc = lbm.BoundaryDict(
            [
                lbm.BounceBackBoundary(
                    "No-Slip Walls", (X == 0) | (X == self.sim.nx - 1) | (Y == 0) | (Y == self.sim.ny - 1)
                ),
            ]
        )
        thermal_bc = lbm.BoundaryDict(
            [
                lbm.BounceBackBoundary("No-Slip Walls", (Y == 0) | (Y == self.sim.ny - 1)),
                lbm.InletBoundary("Left Wall", (X == 0)),
                lbm.InletBoundary("Right Wall", (X == self.sim.nx - 1)),
            ]
        )
        fluid_bc_kwargs = {}

        thermal_bc_kwargs = {
            "Left Wall": {"m": get_wall_temperature(self.sim.y, self.bc_params[0], self.bc_params[1], self.bc_params[2])},
            "Right Wall": {"m": 0.0},
        }
        
        self.sim.set_boundary_conditions(
            (fluid_bc, thermal_bc),
            (fluid_bc_kwargs, thermal_bc_kwargs),
        )

        self.sim_step = 0

        self.tracers = lbm.TracerCollection()

        for _ in range(10):
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
            #self.current_pos = update_tracer(self.sim, self.sim.dfs, self.current_pos)
            #self.target_pos = update_tracer(self.sim, self.sim.dfs, self.target_pos)

        current = self.tracers.get("current")
        print(current[0].x)
        #print(jnp.floor(self.current_pos.x/self.sim.dx).astype(int))

        terminated = self.sim_step > self.sim_steps
        #if jnp.isnan(self.target_pos.obs).any():
        #    print("Simulation crashed")
        #    terminated = True

        observation = 0.0
        reward = 1 if terminated else 0
        info = None
        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            frame = self._render_frame()
            return frame

    def _render_frame(self):
        density, velocity, temperature = self.sim.get_macroscopics(self.sim.dfs)
        x = self.sim.x / self.sim.nx
        y = self.sim.y / self.sim.ny
        img = hv.Image((x, y, temperature[:,:,0].T))
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
        curve = hv.Curve((temperature[0,:], y), "T_0", "y",).opts(
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

