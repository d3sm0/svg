from gym import register
from envs.pendulum import DifferentiablePendulum
from envs.lqg import Lqg

register(id="Lqg-v0", entry_point=Lqg, max_episode_steps=200, reward_threshold=0)

register(id="DifferentiablePendulum-v0", entry_point=DifferentiablePendulum, max_episode_steps=100, reward_threshold=0)
ENV_IDS = {"Lqg-v0", "DifferentiablePendulum-v0"}
