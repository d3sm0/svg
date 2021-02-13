import os

import torch
from gym.wrappers import Monitor

from envs import pendulum_torch
from svg_torch import Policy


def generate_episode(env, policy, gamma=0.99):
    s = env.reset()
    d = False
    while not d:
        mu, _ = policy(s)
        s1, r, d, info = env.step(mu.detach())
        env.render()
        s = s1


log_dir = "logs/13-01-50-19-080985/ckp.pb"
env = pendulum_torch.Pendulum()
env = Monitor(env, os.path.dirname(log_dir))
state_dict = torch.load(log_dir)
policy = Policy(3, 1)
policy.load_state_dict(state_dict)

generate_episode(env, policy)
