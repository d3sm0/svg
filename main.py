import itertools

import experiment_buddy as buddy
import rlego
import torch

import agents
import config
import envs.lqg
from envs.utils import GymWrapper
from models import ActorValue


@torch.no_grad()
def gather_trajectory(env, agent):
    state = env.reset()
    trajectory = rlego.Trajectory()
    while True:
        pi = agent.model.actor(state)
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
        disabled=config.DEBUG,
        extra_modules=["python/3.7", "cuda/11.1/cudnn/8.0"],
    )

    env = GymWrapper(envs.lqg.Lqg())
    model = ActorValue(env.observation_space.shape[0], env.action_space.shape[0], h_dim=config.h_dim).to(config.device)
    writer.watch(model, log="all", log_freq=10)
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
        # ascend the gradient on-policy
        actor_info = agent.optimize_actor(trajectory,
                                          epochs=config.actor_epochs)
        if global_step % config.update_target_every == 0:
            agent.update_target(config.tau)
        writer.add_scalars({
            **env_info,
            **actor_info,
            **critic_info
        }, global_step=global_step)


        # w = agent.model.critic.q[0].weight.T
        # w_mu = agent.model.actor.pi.linear[0].weight.T[:, :1]

        # if not torch.isfinite(actor_info.get("actor/grad_norm")) or not torch.isfinite(critic_info.get("critic/grad_norm")):
        #     print("Found nan in loss", critic_info, actor_info)
        #     break
        # rolling_performance.append(env_info.get("train/return"))
        # avg_return = torch.tensor(rolling_performance)
        # env_info.update({"train/avg_return": avg_return.mean(), "train/avg_std": avg_return.std()})
        # writer.add_scalars({**actor_info, **critic_info, **env_info}, n_samples)

        # n_samples += env_info.get("train/duration")


#         if global_step % config.save_every == 0 and global_step > 0 and config.should_render:
#             render_policy(env, agent)
#             # trajectory, env_return = gather_trajectory(env, agent, replay_buffer)
#
#         if global_step % config.save_every == 0:
#             print(f"Saved at {global_step}. Progress:{n_samples / config.max_steps:.2f}")
#             torch.save(agent, "model.pt")
#


if __name__ == "__main__":
    main()
