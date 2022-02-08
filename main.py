import itertools

import experiment_buddy as buddy
import rlego
import torch

import agents
import config
import envs.lqg
from envs.utils import GymWrapper
from models import ValueZero


@torch.no_grad()
def gather_trajectory(env, agent):
    state = env.reset()
    trajectory = rlego.Trajectory()
    while True:
        pi = agent.model(state)
        action = pi.rsample()
        eps = (action - pi.loc) / pi.scale
        assert torch.linalg.norm(action) < 1e3
        next_state, reward, done, info = env.step(action)
        trajectory.append(rlego.Transition(state, action, reward, next_state, done, eps))
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
        sweep_definition=config.sweep_yaml,
        disabled=False,
        extra_modules=["python/3.7", "cuda/11.1/cudnn/8.0"],
    )

    env = GymWrapper(envs.lqg.Lqg())
    # model = ActorValue(env.observation_space.shape[0], env.action_space.shape[0], h_dim=config.h_dim).to(config.device)
    model = ValueZero(env.observation_space.shape[0], env.action_space.shape[0]).to(config.device)
    # writer.watch(model, log="all", log_freq=10)
    # agent = agents.SVG(model, horizon=config.horizon, dynamics=Dynamics(env.observation_space.shape[0], env.action_space.shape[0]))
    agent = agents.SVG(model)
    run(env, agent, writer)


def run(env, agent, writer):
    for global_step in itertools.count():
        if global_step >= config.max_steps:
            break
        trajectory, env_info = gather_trajectory(env, agent)

        critic_info = agent.optimize_critic(trajectory,
                                            epochs=config.critic_epochs)

        model_info = agent.optimize_model(trajectory, epochs=config.critic_epochs)
        # model_info = {}
        # ascend the gradient on-policy
        actor_info = agent.optimize_actor(trajectory, epochs=config.actor_epochs)
        writer.add_scalars({
            **env_info,
            **actor_info,
            **critic_info,
            **model_info
        }, global_step=global_step)


if __name__ == "__main__":
    main()
