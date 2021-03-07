from gym import register
from gym.envs.classic_control.pendulum import PendulumEnv

from envs.lqg import Lqg
from envs.pendulum_2d import Pendulum2D

# from envs.zika import ZikaEnv

register(id="Lqg-v0", entry_point=Lqg, max_episode_steps=200, reward_threshold=0)

register(id="Pendulum2d-v0", entry_point=Pendulum2D, max_episode_steps=200, reward_threshold=0)
#register(id="Pendulum2d-v0", entry_point=PendulumEnv, max_episode_steps=200, reward_threshold=0)

# register(id="Zika-v0", entry_point=ZikaEnv, max_episode_steps=200, reward_threshold=1e10)

ENV_IDS = {"Lqg-v0", "Pendulum2d-v0", "Zika-v0"}
