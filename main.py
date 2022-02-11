import itertools

import brax.envs
import brax.envs.to_torch
import experiment_buddy as buddy
import rlego
import torch

import agents
import config
from models import ActorValue


@torch.no_grad()
def gather_trajectory(env, agent):
    state = env.reset()
    trajectory = rlego.Trajectory()
    while True:
        pi = agent.model.actor(state)
        action = pi.sample()
        assert torch.linalg.norm(action) < 1e3
        next_state, reward, done, info = env.step(action)
        trajectory.append(rlego.Transition(state, action, reward, next_state, done, torch.cat([pi.loc, pi.scale])))
        state = next_state
        if done:
            break
    return trajectory, info


def main():
    torch.manual_seed(config.seed)
    buddy.register_defaults(config.__dict__)
    writer = buddy.deploy(
        proc_num=config.proc_num,
        host=config.host,
        disabled=config.DEBUG,
        extra_modules=["python/3.7", "cuda/11.1/cudnn/8.0"],
    )

    env = brax.envs.create_gym_env(config.env_id, backend="cpu")
    env = brax.envs.to_torch.JaxToTorchWrapper(env)
    model = ActorValue(env.observation_space.shape[0], env.action_space.shape[0], h_dim=config.h_dim).to(config.device)
    agent = agents.SVGZero(model)
    run(env, agent, writer)


def run(env, agent, writer):
    n_samples = 0

    for global_step in itertools.count():
        if n_samples >= config.max_steps:
            break
        trajectory, env_info = gather_trajectory(env, agent)

        critic_info = agent.optimize_critic(trajectory,
                                            epochs=config.critic_epochs)

        actor_info = agent.optimize_actor(trajectory,
                                          epochs=config.actor_epochs)
        if global_step % config.update_target_every == 0:
            agent.update_target(config.tau)
        writer.add_scalars({
            **env_info,
            **actor_info,
            **critic_info
        }, global_step=global_step)


if __name__ == "__main__":
    main()
