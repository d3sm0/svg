import itertools

import torch
import torch.optim as optim

import config
from envs import torch_envs
from models import Policy, RealDynamics, Dynamics
from svg import generate_episode, train


def scalars_to_tb(writer, scalars, global_step):
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)


def main():
    torch.manual_seed(config.seed)
    env = torch_envs.make_env(config.env_id, horizon=config.horizon)
    policy = Policy(env.env_spec.observation_space, env.env_spec.action_space, h_dim=config.h_dim)
    dynamics = Dynamics(env)
    pi_optim = optim.SGD(list(policy.parameters()), lr=config.policy_lr)
    model_optim = optim.SGD(list(dynamics.parameters()), lr=config.model_lr)
    run(dynamics, env, pi_optim, policy, model_optim)
    env.close()


def run(dynamics, env, pi_optim, agent,model_optim):
    n_samples = 0
    for global_step in itertools.count():
        if n_samples >= config.max_n_samples:
            break
        if global_step % config.save_every == 0:
            print(f"Saved at {global_step}. Progress:{n_samples / config.max_n_samples:.2f}")
            config.tb.add_object("model", agent, global_step)
        trajectory, env_statistics = generate_episode(env, agent)
        scalars_to_tb(config.tb, env_statistics, n_samples)
        agent_loss = train(dynamics, agent, pi_optim, trajectory, model_optim)
        scalars_to_tb(config.tb, agent_loss, n_samples)

        n_samples += len(trajectory)


if __name__ == "__main__":
    main()
