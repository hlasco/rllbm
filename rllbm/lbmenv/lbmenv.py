import os
import functools
from collections import abc

from typing import (
    Any,
    Iterable,
    Mapping,
    Sequence,
)

import gymnasium as gym

import jax
import jax.numpy as jnp
import numpy as np

from rllbm.lbm import Simulation
from rllbm.utils import TracerCollection, update_tracers


__all__ = ["LBMEnv"]


class LBMEnv(gym.Env):
    def __init__(self, config):
        self.config = config

        self.sim = self.init_simulation()
        self.random_key = jax.random.PRNGKey(0)

        self.sim_step = 0
        self.sim_steps_per_env_step = 1
        self.tracers = TracerCollection()

        self.record_video_config = config.get("record_video_config", None)
        self.is_video_enabled = False
        if self.record_video_config is not None:
            self.is_video_enabled = self.record_video_config.get("enabled", False)
        
        if self.is_video_enabled:
            from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
            self.video_id = 0
            self.clip_cls = ImageSequenceClip
            self.fps = self.record_video_config.get("fps", 20)
            self.video_directory = os.path.realpath(
                self.record_video_config.get("directory", ".")
            )
            self.video_frames = []
            self.video_path = None

            os.makedirs(self.video_directory, exist_ok=True)

    def seed(self, seed: int = None):
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")
        self.random_key = jax.random.PRNGKey(seed)

    def add_random_tracers(self, tracer_type: str, num: int, stream: bool = True):
        """Adds a number of random tracers to the environment

        Args:
            tracer_type (str): The type of tracer to add
            num (int): The number of tracers to add
            stream (bool, optional): Whether the tracer should be streamed. Defaults to True.
        """
        self.random_key, key = jax.random.split(self.random_key)
        self.tracers.add_random_tracers(key, tracer_type, self.sim, num, stream)

    def add_grid_tracers(
        self, tracer_type: str, shape: Iterable[int], stream: bool = True
    ):
        """Adds a grid of tracers to the environment

        Args:
            tracer_type (str): The type of tracer to add
            shape (Iterable[int]): The shape of the grid (e.g. (10, 10) for a 10x10 grid)
            stream (bool, optional): Whether the tracer should be streamed. Defaults to True.
        """
        self.tracers.add_grid_tracers(tracer_type, self.sim, shape, stream)

    @property
    def has_tracers(self) -> bool:
        """Checks if the environment has tracers

        Returns:
            bool: True if the environment has tracers, False otherwise
        """
        return len(self.tracers) > 0

    def update_tracers(self) -> None:
        if not self.has_tracers:
            return
        self.tracers = update_tracers(self.sim, self.fluid_state, self.tracers)

    @property
    def fluid_state(self):
        return self.sim.fluid_state

    def check_simulation_crashed(self) -> bool:
        """Checks if the simulation has failed (e.g. NaNs or Infs in the fluid state)
        """
        for state in self.fluid_state:
            if jnp.any(jnp.isnan(state)):
                return True
        return False

    def update_simulation(self, num_steps: int) -> None:
        for _ in range(num_steps):
            self.sim.step()
            self.sim_step += 1
            self.update_tracers()

        self.sim_has_crashed = self.check_simulation_crashed()

    def reset(self, *, seed=None, options=None):
        self.seed(seed)
        self.sim_step = 0
        self.sim_has_crashed = False

        if self.has_tracers:
            self.tracers = TracerCollection()

        self.reset_simulation(seed, options)
        self.update_tracers()

        obs = self.get_observation()
        info = self.get_info()

        return jax_to_numpy(obs), jax_to_numpy(info)

    def step(self, action):
        self.apply_action(action)

        obs = self.get_observation()
        rew = self.get_reward()

        terminated, truncated = self.is_done()

        info = self.get_info()

        if self.is_video_enabled:
            self.video_recording_step(terminated or truncated)
        
        return jax_to_numpy(obs), float(rew), bool(terminated), bool(truncated), jax_to_numpy(info)

    def video_recording_step(self, done):
        self.video_frames.append(self._render_frame())

        if done and (len(self.video_frames) > 1):
            video_id = str(self.video_id).zfill(5)
            env_id = id(self)
            video_name = f"{env_id}_episode-{video_id}.mp4"
            video_path = os.path.join(self.video_directory, video_name)

            clip = self.clip_cls(self.video_frames, fps=self.fps)
            clip.write_videofile(video_path, logger=None)
            
            self.video_path = video_path

            self.video_id += 1
            self.video_frames = []
            
            clip.close()


    def init_simulation(self, *args, **kwargs) -> Simulation:
        """Initializes the simulation"""
        raise NotImplementedError

    def apply_action(self, action) -> None:
        raise NotImplementedError

    def get_observation(self):
        raise NotImplementedError

    def get_reward(self):
        raise NotImplementedError

    def is_done(self):
        raise NotImplementedError

    def _render_frame(self):
        raise NotImplementedError

    def get_info(self):
        return {}


@functools.singledispatch
def jax_to_numpy(value: Any) -> Any:
    """Converts a value to a numpy array."""
    return value

@jax_to_numpy.register(jnp.DeviceArray)
def _devicearray_jax_to_numpy(value: jnp.DeviceArray) -> np.ndarray:
    """Converts a Jax DeviceArray to a numpy array."""
    return np.asarray(value)

@jax_to_numpy.register(jax.Array)
def _array_jax_to_numpy(value: jax.Array) -> np.ndarray:
    """Converts a Jax Array to a numpy array."""
    return np.asarray(value)

@jax_to_numpy.register(abc.Mapping)
def _mapping_jax_to_numpy(
    value: Mapping[str, jnp.DeviceArray | Any]
) -> Mapping[str, np.ndarray | Any]:
    """Converts a dictionary of Jax DeviceArrays to a mapping of numpy arrays."""
    return type(value)(**{k: jax_to_numpy(v) for k, v in value.items()})

@jax_to_numpy.register(abc.Sequence)
def _sequence_jax_to_numpy(
    value: Sequence[np.ndarray | Any],
) -> Sequence[jnp.DeviceArray | Any]:
    """Converts a Sequence of Jax arrays to a Sequence of numpy arrays."""
    return type(value)(jax_to_numpy(v) for v in value)