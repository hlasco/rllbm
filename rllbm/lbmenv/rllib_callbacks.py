from typing import Dict
from ray.rllib import Policy, BaseEnv
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.utils.typing import PolicyID
from wandb import Video

__all__ = ["VideoCallback"]

class VideoCallback(DefaultCallbacks):
    def on_episode_end(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[str, Policy],
            episode: Episode,
            env_index: int,
            **kwargs,
    ) -> None:
        
        envs = base_env.get_sub_environments()

        if not hasattr(envs[env_index], "video_path"):
            return

        video_path = envs[env_index].video_path
        if video_path is not None:
            episode.media["video"] = Video(video_path)
            envs[env_index].video_path = None