import itertools
import os
from datetime import datetime
from types import SimpleNamespace

import torch
import torch.optim as optim
import torch.utils.tensorboard as tb

from envs import torch_envs
from models import Policy, Dynamics
from svg_torch import generate_episode, train, train_model_on_traj

config = SimpleNamespace(use_oracle=False,
                         initial_steps=int(1e4),  # int(1e4),
                         policy_lr=1e-4,
                         model_lr=1e-3,
                         model_std=0.1,
                         reward_lr=1e-3,
                         batch_size=64,
                         train_on_buffer=True,
                         learn_reward=True,
                         shuffle=True,
                         horizon=100,
                         opt_steps=1,
                         max_n_samples=int(1e6),
                         env_id="Pendulum2d-v0",
                         seed=0,
                         gamma=0.99,
                         save_every=100,
                         )


def extend(writer, scalars, global_step):
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)


def main():
    torch.manual_seed(config.seed)

    env = torch_envs.make_env(config.env_id, horizon=config.horizon)
    policy = Policy(env.env_spec.observation_space, env.env_spec.action_space, h_dim=16)

    dynamics = Dynamics(env, learn_reward=config.learn_reward, std=config.model_std)
    # dynamics = RealDynamics(env)

    pi_optim = optim.SGD(policy.parameters(), lr=config.policy_lr)
    model_optim = optim.SGD(dynamics.parameters(), lr=config.model_lr)

    # if config.leanr_reward:
    #    reward_optim = optim.SGD(dynamics.r.parameters(), lr=config.reward_lr)

    dtm = datetime.now().strftime("%d-%H-%M-%S-%f")
    writer = tb.SummaryWriter(log_dir=f"logs/{dtm}")
    run(dynamics, env, model_optim, pi_optim, policy, writer)
    writer.close()


def run(dynamics, env, model_optim, pi_optim, agent, writer):
    # number of frames
    n_samples = 0
    for global_step in itertools.count():
        if n_samples >= config.max_n_samples:
            break
        if global_step % config.save_every == 0:
            print(f"Saved at {global_step}. Progress:{n_samples / config.max_n_samples:.2f}")
            torch.save(agent, os.path.join(writer.log_dir, f"latest.pt"))
        trajectory, env_statistics = generate_episode(env, agent)
        extend(writer, env_statistics, n_samples)
        # extend(writer, trajectory.get_statistics(), n_samples)

        s, a, *_ = list(zip(*trajectory._data))
        s = torch.stack(s).norm(dim=-1).mean()
        a = torch.stack(a).norm(dim=-1).mean()
        writer.add_scalar("env/state_norm", s, global_step=n_samples)
        writer.add_scalar("env/action_norm", a, global_step=n_samples)

        # if not config.use_oracle:
        # if config.learn_reward or config.train_on_buffer:
        # buffer.add_trajectory(trajectory)

        # if config.train_on_buffer:
        #    model_loss = train_model_on_buffer(dynamics, buffer, model_optim)
        # else:64
        model_loss = train_model_on_traj(trajectory, dynamics, model_optim, batch_size=config.batch_size,
                                         shuffle=config.shuffle)
        extend(writer, model_loss, n_samples)

        # if config.learn_reward:
        #    reward_loss = train_reward_on_buffer(dynamics, buffer, reward_optim)
        #    writer.add_scalar("train/reward_loss", reward_loss, global_step=n_samples)
        # writer.add_scalar("train/model_loss", model_loss, global_step=n_samples)

        if n_samples > config.initial_steps:
            agent_loss = train(dynamics, agent, pi_optim, trajectory, gamma=env.gamma, opt_steps=config.opt_steps)
            extend(writer, agent_loss, n_samples)

        # writer.add_scalar("train/return", ret, global_step=n_samples)
        # writer.add_scalar("train/grad_norm", grad_norm, global_step=n_samples)
        n_samples += len(trajectory)


if __name__ == "__main__":
    main()
