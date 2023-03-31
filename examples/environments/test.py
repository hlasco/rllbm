import comet_ml
from comet_ml.integration.gymnasium import CometLogger
import gymnasium as gym

from rllbm.environments.env import LBMEnv
import time

env = LBMEnv(render_mode="rgb_array")

def episode_trigger(ep_id):
    return ep_id % 1 == 0

env = gym.wrappers.RecordVideo(env, 'visu', episode_trigger=episode_trigger)

experiment = comet_ml.Experiment(project_name="rllbm-thermal-fluid-control")
env = CometLogger(env, experiment)

for x in range(2):
    
    print(x)

    observation, info = env.reset()
    truncated = False
    terminated = False
    i=0
    while not (truncated or terminated):
        start = time.time()
        observation, reward, terminated, truncated, info = env.step(
            env.action_space.sample()
        )
        #print("step:", i, "time:", time.time() - start)
        i+=1

env.close()

experiment.end()