import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from ray.rllib.algorithms.sac import SACConfig, RNNSACConfig
from ray.tune.registry import register_env
from ray import air
from ray import tune

import copy

from ray.air.integrations.wandb import WandbLoggerCallback
from rllbm.lbmenv import VideoCallback

from heat_wall_env import HeatWallEnv, HeatWallEnvConfig

env_config = copy.copy(HeatWallEnvConfig)

def make_env(cfg):
    
    env = HeatWallEnv(cfg)
    return env

def make_eval_env(cfg):
    eval_config = copy.deepcopy(cfg)
    eval_config["record_video_config"]["enabled"] = True
    eval_config["record_video_config"]["directory"] = "./evaluation"
    env = make_env(eval_config)
    return env

register_env("train_env", make_env)
register_env("eval_env", make_eval_env)

config = (
    SACConfig()

    .environment(
        env="train_env",
        env_config=env_config,
    )

    .debugging(
        log_level='INFO'
    )

    .training(
        train_batch_size=32,
        training_intensity=32,
        gamma=0.995,
        optimization_config = {
            "actor_learning_rate": 1e-4,
            "critic_learning_rate": 1e-4,
            "entropy_learning_rate": 1e-4,
        },
        q_model_config={
            "fcnet_hiddens": [256, 256],
            "post_fcnet_hiddens": [],
            "post_fcnet_activation": None,
        },
        policy_model_config={
            "fcnet_hiddens": [256, 256],
            "post_fcnet_hiddens": [],
            "post_fcnet_activation": None,
        },
        replay_buffer_config={
            "type": "MultiAgentPrioritizedReplayBuffer",
            "capacity": int(1e6),
            # If True prioritized replay buffer will be used.
            "prioritized_replay": True,
            "prioritized_replay_alpha": 0.6,
            "prioritized_replay_beta": 0.4,
            "prioritized_replay_eps": 1e-6,
        },
        num_steps_sampled_before_learning_starts=0,
    )

    .resources(
        num_gpus=1.0,
    )

    .rollouts(
        remote_worker_envs=True,
        num_rollout_workers=0,
        num_envs_per_worker=64,
    )

    .framework('torch')

    .evaluation(
        evaluation_interval=40,
        evaluation_duration=1,
        evaluation_parallel_to_training=True,
        evaluation_num_workers=1,
        evaluation_sample_timeout_s=None,
        evaluation_config=SACConfig.overrides(
            env="eval_env",
            num_envs_per_worker=1,
            remote_worker_envs=False,
        )
    )

    .reporting(
        min_sample_timesteps_per_iteration=1000,
    )

    .callbacks(
        VideoCallback
    )
)

tune.Tuner(  
    "SAC",
    run_config=air.RunConfig(
        verbose=1,
        stop={"num_env_steps_sampled": int(1e6)},
        callbacks=[
            WandbLoggerCallback(project="Heat-Wall-v0"),
        ]
    ),
    param_space=config.to_dict(),
).fit()
