
from comet_ml.integration.gymnasium import CometLogger
import gymnasium as gym

from rllbm.environment import ThermalFluidControl
import time

env = ThermalFluidControl(render_mode="rgb_array")

def episode_trigger(ep_id):
    return ep_id % 1 == 0

env = gym.wrappers.RecordVideo(env, 'visu-rllbm-thermal-fluid-control', episode_trigger=episode_trigger)

try:
    import comet_ml
    experiment = comet_ml.Experiment(project_name="rllbm-thermal-fluid-control")
    env = CometLogger(env, experiment)
    comet_logging = True
except Exception:
    comet_logging = False

for x in range(2):
    observation, info = env.reset()
    truncated = False
    terminated = False
    i=0
    while not (truncated or terminated):
        start = time.time()
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(
            action
        )
        print("step:", i, "time:", time.time() - start)
        i+=1

env.close()

if comet_logging:
    experiment.end()