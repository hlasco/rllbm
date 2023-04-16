import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from rllbm.lbm import (
    Simulation,
    Domain,
    D2Q5,
    D2Q9,
    ThermalFluidLattice,
    BounceBackBoundary,
    InletBoundary
)
from rllbm.lbmenv import LBMEnv

import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import LogFormatter 


__all__ = ["HeatWallEnv", "HeatWallEnvConfig"]


HeatWallEnvConfig = {
    "nx": 64,
    "ny": 64,
    "temperature_amplitude": 1.0,
    "run_time": 500.0,
    "step_time": 1.0,
    "prandtl": 0.71,
    "rayleigh": 1e8,
    "buoyancy": 0.001,
    "record_video_config":{
        "enabled": False,
        "fps": 20,
        "directory": "./test_env",
    }
}

class HeatWallEnv(LBMEnv):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, config=HeatWallEnvConfig, render_mode="rgb_array"):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        super().__init__(config)

        convection_timescale = 1.0

        run_time = self.config["run_time"] * convection_timescale
        step_time = self.config["step_time"] * convection_timescale

        dt = (self.config["buoyancy"] * self.sim.dx)**0.5

        self.sim_steps = int(run_time / dt)
        self.sim_steps_per_env_step = int(step_time / dt)

        self.bc_params_min = jnp.array([0.0, -0.5 * self.config["temperature_amplitude"], 0.01])
        self.bc_params_max = jnp.array([1.0, 0.5 * self.config["temperature_amplitude"], 0.1])

        self.observation_space = gym.spaces.Box(low=-2, high=2, shape=(15,))
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,))

    def init_simulation(self) -> Simulation:
        """Initialize the simulation."""

        # Define the domain
        domain = Domain(shape=(self.config["nx"], self.config["ny"]), bounds=(0.0, 1.0, 0.0, 1.0))

        # Define relaxation parameters
        dx = domain.dx
        dt = (self.config["buoyancy"] * dx)**0.5

        viscosity = (self.config["prandtl"] / self.config["rayleigh"]) ** 0.5 * dt / dx**2
        kappa = viscosity / self.config["prandtl"]
        omegas = {
            "FluidLattice": 1.0 / (3 * viscosity + 0.5),
            "ThermalLattice": 1.0 / (3 * kappa + 0.5),
        }

        lattice = ThermalFluidLattice(
            fluid_stencil=D2Q9,
            thermal_stencil=D2Q5,
            buoyancy=jnp.array([0, self.config["buoyancy"]]),
        )

        sim = Simulation(domain, lattice, omegas)

        # Set the boundary conditions
        sim.set_boundary_conditions(
            BounceBackBoundary(
                "walls", sim.bottom | sim.top | sim.left | sim.right
            ),
            "FluidLattice",
        )

        sim.set_boundary_conditions(
            BounceBackBoundary("bottom", sim.bottom), "ThermalLattice"
        )
        sim.set_boundary_conditions(
            BounceBackBoundary("top", sim.top), "ThermalLattice"
        )
        sim.set_boundary_conditions(
            BounceBackBoundary("right", sim.right), "ThermalLattice"
        )
        sim.set_boundary_conditions(
            InletBoundary("left", sim.left), "ThermalLattice"
        )


        sim.update_boundary_condition("left", {"m": 0.0}, "ThermalLattice")

        return sim

    def reset_simulation(self, seed, options):
        nx, ny = self.config["nx"], self.config["ny"]
        self.sim.set_initial_conditions(
            rho=jnp.ones((nx, ny, 1)),
            T=jnp.zeros((nx, ny, 1)),
            u=jnp.zeros((nx, ny, 2)),
        )

        self.bc_params = jax.random.uniform(
            self.random_key,
            shape=(3,),
            minval=self.bc_params_min,
            maxval=self.bc_params_max,
        )

        self.add_random_tracers("current", 1, stream=True)
        self.add_random_tracers("target", 1, stream=False)

    def apply_action(self, action):

        bc_params_shift = (self.bc_params_max - self.bc_params_min) / 10 * action
        new_bc_params = self.bc_params + bc_params_shift

        self.bc_params = jnp.clip(new_bc_params, self.bc_params_min, self.bc_params_max)

        wall_tenperature = get_wall_temperature(self.sim.y, self.bc_params)
        self.sim.update_boundary_condition("left", {"m": wall_tenperature}, "ThermalLattice")

        self.update_simulation(self.sim_steps_per_env_step)


    def get_observation(self):
        target = self.tracers.get("target")[0]
        current = self.tracers.get("current")[0]

        obs = jnp.concatenate(
            [self.bc_params, target.flatten(), current.flatten()]
        )

        if self.sim_has_crashed:
            return jnp.ones_like(obs)
        
        return obs

    def get_reward(self):
        if self.sim_has_crashed:
            return -jnp.sqrt(2)
        
        target = self.tracers.get("target")[0]
        current = self.tracers.get("current")[0]

        distance = jnp.linalg.norm(target.x-current.x, ord=2)

        return -distance

    def is_done(self):
        truncated = self.sim_has_crashed
        terminated = self.sim_step > self.sim_steps
        
        return terminated, truncated

    def render(self):
        if self.render_mode == "rgb_array":
            frame = self._render_frame()
            return frame

    def _render_frame(self):
        fig, (ax0, ax1) = plt.subplots(ncols=2, sharey=True, figsize=(8, 4))

        x = self.sim.x
        y = self.sim.y

        temperature_max = 0.5 * self.config["temperature_amplitude"]
        

        img = ax1.imshow(
            self.fluid_state.T[:, :, 0].T,
            extent=[x.min(), x.max(), y.min(), y.max()],
            cmap="RdBu_r",
            norm=SymLogNorm(0.01*temperature_max, vmin=-temperature_max, vmax=temperature_max),
            origin="lower",
        )
        ax1.set_xlabel("x")

        formatter = LogFormatter(10, labelOnlyBase=False)

        cbar = fig.colorbar(img, ax=ax1, format=formatter )
        cbar.set_label("Fluid Temperature")

        curve = ax0.plot(
            self.fluid_state.T[0, :],
            y,
            label="T_0",
            color="blue",
        )
        
        ax0.set_xlabel("Wall Temperature")
        ax0.set_xlim(-temperature_max, temperature_max)
        ax0.set_ylim(y.min(), y.max())
        ax0.set_ylabel("y")
        ax0.set_aspect(4.0)
        ax0.xaxis.set_label_coords(0.5, -0.1)

        current = self.tracers.get("current")
        current = np.stack([t.x for t in current], axis=0)
        ax1.scatter(current[:, 0], current[:, 1], label="Current", marker="o", color="#7C5104", alpha=0.8)

        target = self.tracers.get("target")
        target = np.stack([t.x for t in target], axis=0)
        ax1.scatter(target[:, 0], target[:, 1], label="Target", marker="o", color="#2ECC71", alpha=0.8)

        ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                      ncols=2, mode="expand", borderaxespad=0.)
        fig.tight_layout(pad=2)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape((h, w, 3))
        
        plt.close(fig)
        return frame

@jax.jit
def get_wall_temperature(x, bc_params):
    """ """
    x_peak, amp, width = bc_params[0], bc_params[1], bc_params[2]
    x_min = x[0]
    x_max = x[-1]

    x_ = (x - x_min) / (x_max - x_min)

    ret = amp * jnp.exp(-0.5 * ((x_ - x_peak) / (width)) ** 2) * jnp.sin(x_ * jnp.pi)
    return ret

if __name__ == "__main__":
    import copy
    env_config = copy.copy(HeatWallEnvConfig)
    env_config["record_video_config"]["enabled"] = False
    env = HeatWallEnv(env_config)

    observation, info = env.reset()
    truncated = False
    terminated = False
    while not (truncated or terminated):
        observation, reward, terminated, truncated, info = env.step(
            env.action_space.sample()
        )
    env.close()