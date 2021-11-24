import collections
import os

import torch
from gym.wrappers import Monitor


def generate_episode(env, policy, max_steps=200):
    state = env.reset()
    d = False
    info = {}
    total_return = 0
    for _ in range(max_steps):
        mu, _ = policy(torch.tensor(state, dtype=torch.float32))
        state, r, d, info = env.step(mu.detach())
        total_return += r
        env.render()
    env.close()
    return {"total_return": total_return}


def eval_policy(env, agent, log_dir, eval_runs=1):
    agent.eval()
    env.seed(0)
    env = Monitor(env, os.path.dirname(log_dir), force=True)
    for _ in range(eval_runs):
        info = generate_episode(env, agent)
    env.close()
    del env
    return info
