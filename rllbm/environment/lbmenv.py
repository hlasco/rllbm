import abc
import os
from collections.abc import Iterable
import gymnasium as gym

import jax
import jax.numpy as jnp
from rllbm.lbm import Simulation, TracerCollection

__all__ = ["LBMEnv"]

class LBMEnv(gym.Env, abc.ABC):    
    def __init__(self, params):
        self.params = params
        
        self.sim = self.initialize_sim()
        
        if not self.sim.is_initialized:
            raise ValueError(
                "Simulation is not initialized. This is likely due to the initialize_sim() method."
            )
        
        self.sim_step = 0
        self.tracers = TracerCollection()
    
    @abc.abstractmethod
    def initialize_sim(self, *args, **kwargs) -> Simulation:
        """Initializes the simulation
        """
        pass
    
    def __getattr__(self, name):
        try:
            return self.params[name]
        except KeyError:
            # check if a default value was provided for the missing attribute
            if hasattr(type(self), name):
                return getattr(type(self), name)
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
    @property
    def has_tracers(self) -> bool:
        """Checks if the environment has tracers

        Returns:
            bool: True if the environment has tracers, False otherwise
        """
        return len(self.tracers) > 0
        
    def simulation_failed(self) -> bool:
        """Checks if the simulation has failed (e.g. NaNs or Infs in the fluid state)

        Returns:
            bool: True if the simulation has failed, False otherwise
        """
        fluid_state = self.sim.get_macroscopics(self.sim.dfs)
        for state in fluid_state:
            if jnp.any(jnp.isnan(state)):
                return True
            if jnp.any(jnp.isinf(state)):
                return True
        return False
    
    def simulation_step(self) -> None:
        """Performs a single simulation step
        """
        self.sim.step()
    
    def reset(self, seed, initial_condition, reset_tracers=True):
        if seed is None:
            seed = int.from_bytes(os.urandom(4), 'big')
        self.sim_step = 0
        self.random_key = jax.random.PRNGKey(seed)
        if self.has_tracers and reset_tracers:
            self.tracers = TracerCollection()
        
        self.sim.dfs = initial_condition
        
    def add_random_tracers(self, tracer_type: str, num: int, stream: bool = True):
        """Adds a number of random tracers to the environment

        Args:
            tracer_type (str): The type of tracer to add
            num (int): The number of tracers to add
            stream (bool, optional): Whether the tracer should be streamed. Defaults to True.
        """
        self.random_key, key = jax.random.split(self.random_key)
        self.tracers.add_random_tracers(key, tracer_type, self.sim, num, stream)
        
    def add_grid_tracers(self, tracer_type: str, shape: Iterable[int], stream: bool = True):
        """Adds a grid of tracers to the environment

        Args:
            tracer_type (str): The type of tracer to add
            shape (Iterable[int]): The shape of the grid (e.g. (10, 10) for a 10x10 grid)
            stream (bool, optional): Whether the tracer should be streamed. Defaults to True.
        """
        self.tracers.add_grid_tracers(tracer_type, self.sim, shape, stream)