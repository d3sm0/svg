import collections
import itertools

import experiment_buddy as buddy
import torch
import torch.optim as optim
from brax.envs import to_torch, create_gym_env

import agents
import config
import svg
from buffer import Trajectory, Transition, Buffer
from models import ActorValue


# this should go in buddy
def scalars_to_tb(writer, scalars, global_step):
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)


def gather_trajectory(env, agent, gamma=0.99):
    state = env.reset()
    trajectory = Trajectory()
    total_return = 0
    t = 0
    while True:
        action, eps = agent.get_action(state)
        assert torch.linalg.norm(action) < 1e3
        next_state, reward, done, _ = env.step(action)
        trajectory.append(Transition(state, action, reward, next_state, done, eps))
        state = next_state
        total_return += reward * gamma**t
        t += 1
        if done:
            break
    return trajectory, {"train/return": total_return, "train/duration": t}


def main():
    torch.manual_seed(config.seed)
    buddy.register_defaults(config.__dict__)
    tb = buddy.deploy(
        proc_num=config.proc_num,
        host=config.host,
        sweep_yaml=config.sweep_yaml,
        disabled=config.DEBUG,
        wandb_kwargs=dict(entity="ihvg"),
        extra_modules=["python/3.7", "cuda/11.1/cudnn/8.0"],
    )

    env = create_gym_env(config.env_id)
    env = to_torch.JaxToTorchWrapper(env, device=config.device)
    agent = ActorValue(env.observation_space.shape[0], env.action_space.shape[0], h_dim=config.h_dim).to(config.device)
    agent = agents.SVGZero(agent, horizon=config.train_horizon)
    actor_optim = optim.Adam(agent.actor.parameters(), lr=config.policy_lr)
    critic_optim = optim.Adam(agent.critic.parameters(), lr=config.critic_lr)
    run(env, agent, actor_optim, critic_optim, tb)


def render_policy(env, agent):
    state = env.reset(0)
    while not state.done:
        action, _ = agent.get_action(state)
        state, _ = env.step(action)
        # env.render(state.obs)
    env.close()


def run(env, agent, actor_optim, critic_optim, tb):
    n_samples = 0
    replay_buffer = Buffer(config.buffer_size)
    rolling_performance = collections.deque(maxlen=20)
    rolling_performance.append(torch.zeros((1,)))

    for global_step in itertools.count():
        if n_samples >= config.max_steps:
            break
        trajectory, env_info = gather_trajectory(env, agent, gamma=config.gamma)
        replay_buffer.extend(trajectory)
        # keep a critic "off-policy"
        critic_info = svg.optimize_critic(replay_buffer,
                                          agent,
                                          critic_optim,
                                          batch_size=config.batch_size,
                                          epochs=config.critic_epochs)
        # ascend the gradient on-policy
        actor_info = svg.optimize_actor(trajectory,
                                        agent,
                                        env,
                                        actor_optim,
                                        batch_size=config.batch_size,
                                        epochs=config.actor_epochs)

        if global_step % config.update_target_every == 0:
            agent.update_target(config.tau)
        print(global_step, critic_info, actor_info)
        if not torch.isfinite(actor_info.get("actor/grad_norm")) or not torch.isfinite(
                critic_info.get("critic/grad_norm")):
            print("Found nan in loss", critic_info, actor_info)
            break
        rolling_performance.append(env_info.get("train/return"))
        avg_return = torch.tensor(rolling_performance)
        env_info.update({"train/avg_return": avg_return.mean(), "train/avg_std": avg_return.std()})
        scalars_to_tb(tb, {**actor_info, **critic_info, **env_info}, n_samples)

        n_samples += env_info.get("train/duration")

        if global_step % config.save_every == 0 and global_step > 0 and config.should_render:
            render_policy(env, agent)
            # trajectory, env_return = gather_trajectory(env, agent, replay_buffer)

        if global_step % config.save_every == 0:
            print(f"Saved at {global_step}. Progress:{n_samples / config.max_steps:.2f}")
            torch.save(agent, "model.pt")


if __name__ == "__main__":
    main()
