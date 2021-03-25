import collections
from envs import torch_envs
import config
import os

import torch
from gym.wrappers import  Monitor


def generate_episode(env, policy):
    s, *_ = env.reset()
    d = False
    info = {}
    while not d:
        mu, _ = policy(s)
        s1, r, d, info = env.step(mu.detach())
        env.render()
        s = s1
    env.close()
    return info


def eval_policy(log_dir, eval_runs=1):
    env = torch_envs.make_env(config.env_id)
    env = Monitor(env, os.path.dirname(log_dir), force=True)
    policy = torch.load(log_dir)
    agg_info = collections.defaultdict(lambda: 0)
    for _ in range(eval_runs):
        info = generate_episode(env, policy)
        for k in info.keys():
            agg_info[f"eval/{k}"] += info[k] / eval_runs
    env.close()
    del env
    del policy
    return dict(agg_info)


#if __name__ == '__main__':
#    import argparse
#
#    parser = argparse.ArgumentParser("eval")
#    parser.add_argument("--logdir", type=str, required=True)
#log_dir = "runs/objects/local_Mar24_18-17-18/model-2200.pt"
log_dir = "runs/objects/less_hp_Mar25_02-19-12/model-9400.pt"
#log_dir = "runs/objects/no_id_Mar25_02-07-29/model-4400.pt"
#log_dir ="runs/objects/asfd_Mar25_01-41-54/model-5300.pt"
eval_policy(log_dir)
