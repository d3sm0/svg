import itertools

import experiment_buddy as buddy
import torch
import torch.optim as optim

import agents
import config
import svg
from buffer import Trajectory, Transition, Buffer
from envs.pendulum import Pendulum
# from envs.cartpole import CartPole
from models import ActorCritc, ActorValue


# this should go in buddy
def scalars_to_tb(writer, scalars, global_step):
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)


def gather_trajectory(env, agent, gamma=0.99):
    state = env.reset(0)
    trajectory = Trajectory()
    total_return = 0
    t = 0
    while not state.done:
        action, eps = agent.get_action(state.obs)
        next_state = env.step(state, action)
        trajectory.append(Transition(state.obs, action, next_state.reward, next_state.obs, next_state.done, eps))
        state = next_state
        total_return += state.reward * gamma ** t
        t += 1
    return trajectory, {"return": total_return, "duration": t}


def main():
    torch.manual_seed(config.seed)
    buddy.register_defaults(config.__dict__)
    tb = buddy.deploy(proc_num=config.proc_num,
                      host=config.host,
                      sweep_yaml=config.sweep_yaml,
                      disabled=config.DEBUG,
                      wandb_kwargs=dict(entity="ihvg"))
    env = Pendulum(horizon=config.horizon)  # agent follows brax convention
    agent = ActorValue(env.observation_size, env.action_size, h_dim=config.h_dim)
    agent = agents.SVGZero(agent, horizon=config.train_horizon)
    actor_optim = optim.Adam(agent.actor.parameters(), lr=config.policy_lr)
    critic_optim = optim.Adam(agent.critic.parameters(), lr=config.critic_lr)
    run(env, agent, actor_optim, critic_optim, tb)


def render_policy(env, agent):
    state = env.reset(0)
    while not state.done:
        action, _ = agent.get_action(state.obs)
        state = env.step(state, action)
        env.render(state.obs)
    env.close()


def run(env, agent, actor_optim, critic_optim, tb):
    n_samples = 0
    replay_buffer = Buffer(config.buffer_size)
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
        actor_info = svg.optimize_actor(replay_buffer,
                                        agent,
                                        env,
                                        actor_optim,
                                        batch_size=config.batch_size,
                                        epochs=config.actor_epochs)

        if global_step % config.update_target_every == 0:
            agent.update_target(config.tau)

        if not torch.isfinite(actor_info.get("actor/grad_norm")) or not torch.isfinite(critic_info.get("critic/grad_norm")):
            print("Found nan in loss")
            break
        scalars_to_tb(tb, {**actor_info, **critic_info, **env_info},n_samples)
        n_samples += env_info.get("duration")

        if global_step % config.save_every == 0 and global_step > 0 and config.should_render:
            render_policy(env, agent)
            # trajectory, env_return = gather_trajectory(env, agent, replay_buffer)

        if global_step % config.save_every == 0:
            print(f"Saved at {global_step}. Progress:{n_samples / config.max_steps:.2f}")
        torch.save(agent.critic, "critic")


if __name__ == "__main__":
    main()
