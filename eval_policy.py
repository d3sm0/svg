import os

import torch
from gym.wrappers import Monitor

from envs import pendulum_2d
from models import Policy


def generate_episode(env, policy, gamma=0.99):
    s = env.reset()
    d = False
    while not d:
        mu, _ = policy(s)
        s1, r, d, info = env.step(mu.detach())
        env.render()
        s = s1


#log_dir = "logs/07-17-57-11-940048/latest.pt"
log_dir = "logs/07-18-03-07-188148/latest.pt"
env = pendulum_2d.Pendulum2D()
env = Monitor(env, os.path.dirname(log_dir), force=True)
policy = torch.load(log_dir)
#policy = Policy(3, 1)
#policy.load_state_dict(state_dict)

generate_episode(env, policy)
