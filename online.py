import collections
import itertools

import experiment_buddy as buddy
import gym
import rlego
import torch
import torch.optim as optim

import agents
import config
import svg
from env.utils import GymWrapper
from models import ActorValue


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

    # $ env = create_gym_env(config.env_id)
    # $ env = to_torch.JaxToTorchWrapper(env, device=config.device)
    env = GymWrapper(gym.make("Pendulum-v1"))
    agent = ActorValue(env.observation_space.shape[0], env.action_space.shape[0], h_dim=config.h_dim).to(config.device)
    agent = agents.SVGZero(agent, horizon=config.train_horizon)
    actor_optim = optim.Adam(agent.actor.parameters(), lr=config.policy_lr)
    critic_optim = optim.Adam(agent.critic.parameters(), lr=config.critic_lr)
    run(env, agent, actor_optim, critic_optim, tb)


def run(env, agent, actor_optim, critic_optim, tb):
    replay_buffer = rlego.Buffer(config.buffer_size)
    rolling_performance = collections.deque(maxlen=20)
    rolling_performance.append(torch.zeros((1,)))

    state = env.reset()
    n_samples = 0
    total_return = 0
    t = 0
    for global_step in itertools.count():
        if n_samples >= config.max_steps:
            break
        action, eps = agent.get_action(state.unsqueeze(0))
        assert torch.linalg.norm(action) < 1e3
        next_state, reward, done, _ = env.step(action.numpy())
        replay_buffer.append(rlego.Transition(state, action, reward, next_state, done, eps))
        state = next_state
        total_return += reward * config.gamma ** t
        t += 1
        n_samples += 1
        if done:
            rolling_performance.append(total_return)
            state = env.reset()
            t = 0
            total_return = 0
        critic_info = svg.optimize_critic(replay_buffer,
                                          agent,
                                          critic_optim,
                                          batch_size=config.batch_size,
                                          epochs=config.critic_epochs)
        # ascend the gradient on-policy
        actor_info = svg.optimize_actor(replay_buffer,
                                        agent,
                                        env,
                                        actor_optim,
                                        batch_size=config.batch_size,
                                        epochs=config.actor_epochs)

        if global_step % config.update_target_every == 0:
            agent.update_target(config.tau)
        if not torch.isfinite(actor_info.get("actor/grad_norm")) or not torch.isfinite(
                critic_info.get("critic/grad_norm")):
            print("Found nan in loss", critic_info, actor_info)
            break
        avg_return = torch.tensor(rolling_performance)
        tb.add_scalars(
            {**actor_info, **critic_info, "train/avg_return": avg_return.mean(), "train/avg_std": avg_return.std()},
            n_samples)


if __name__ == "__main__":
    main()
