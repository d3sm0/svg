import itertools
from types import SimpleNamespace

import torch
import torch.optim as optim

from buffer import Buffer
from envs import torch_envs
from models import Policy, Dynamics, RealDynamics
from svg_torch import generate_episode, train, train_model_on_traj
import config


def scalars_to_tb(writer, scalars, global_step):
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)


def main():
    torch.manual_seed(config.seed)

    env = torch_envs.make_env(config.env_id, horizon=config.horizon)
    policy = Policy(env.env_spec.observation_space, env.env_spec.action_space, h_dim=config.h_dim)

    # dynamics = Dynamics(env, learn_reward=agent_config.learn_reward, std=agent_config.model_std)
    dynamics = RealDynamics(env)

    pi_optim = optim.SGD(policy.parameters(), lr=config.policy_lr)
    model_optim = None  # optim.SGD(dynamics.parameters(), lr=agent_config.model_lr)

    writer = config.tb
    run(dynamics, env, model_optim, pi_optim, policy, writer)
    writer.close()


def run(dynamics, env, model_optim, pi_optim, agent, writer):
    # number of frames
    n_samples = 0
    buffer = Buffer()
    for global_step in itertools.count():
        if n_samples >= config.max_n_samples:
            break
        if global_step % config.save_every == 0:
            print(f"Saved at {global_step}. Progress:{n_samples / config.max_n_samples:.2f}")
            config.tb.add_object("model", agent, global_step)
        trajectory, env_statistics = generate_episode(env, agent)
        buffer.add(trajectory)
        scalars_to_tb(writer, env_statistics, n_samples)

        s, a, *_ = list(zip(*trajectory._data))
        s = torch.stack(s).norm(dim=-1).mean()
        a = torch.stack(a).norm(dim=-1).mean()
        writer.add_scalar("env/state_norm", s, global_step=n_samples)
        writer.add_scalar("env/action_norm", a, global_step=n_samples)

        agent_loss = train(dynamics, agent, pi_optim, trajectory, gamma=env.gamma)
        scalars_to_tb(writer, agent_loss, n_samples)

        n_samples += len(trajectory)


if __name__ == "__main__":
    main()
