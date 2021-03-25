import collections
from envs import jax_envs
import config
import os
import flax
import flax.training.checkpoints as ckpts

# import torch
from gym.wrappers import Monitor


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
    env = jax_envs.make_env(config.env_id)
    env = Monitor(env, os.path.dirname(log_dir), force=True)
    policy = ckpts.restore_checkpoint(ckpt_dir=log_dir, target=None)
    agg_info = collections.defaultdict(lambda: 0)
    for _ in range(eval_runs):
        info = generate_episode(env, policy)
        for k in info.keys():
            agg_info[f"eval/{k}"] += info[k] / eval_runs
    env.close()
    del env
    del policy
    return dict(agg_info)


# if __name__ == '__main__':
#    import argparse
#
#    parser = argparse.ArgumentParser("eval")
#    parser.add_argument("--logdir", type=str, required=True)
log_dir = "runs/objects/local_Mar24_18-17-18/model-2200.pt"
eval_policy(log_dir)
