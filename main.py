import itertools

import torch
import torch.optim as optim
import config
import experiment_buddy as buddy

from eval_policy import eval_policy
from models import Agent
from envs.pendulum import Pendulum
import svg
import utils



from buffer import Trajectory, Transition
from gym.envs.classic_control import PendulumEnv

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
        action,eps = agent.get_action(state.obs)
        next_state = env.step(state, action)
        trajectory.append(Transition(state.obs, action, next_state.reward, next_state.obs, next_state.done, eps))
        # replay_buffer.append(Transition(state.obs, action, next_state.reward, next_state.obs, next_state.done, eps))
        state = next_state
        total_return += state.reward * gamma**t
        t+=1
    return trajectory, total_return


def main():
    torch.manual_seed(config.seed)
    buddy.register_defaults(config.__dict__)
    tb = buddy.deploy(disabled=True)
    env = Pendulum() # agent follows brax convention
    agent = Agent(env.observation_size, env.action_size, h_dim=config.h_dim)
    actor_optim = optim.Adam(agent.actor.parameters(), lr=config.policy_lr)
    critic_optim= optim.Adam(agent.critic.parameters(), lr=config.policy_lr)
    run(env, agent, actor_optim,critic_optim,tb)


def run(env, agent, actor_optim ,critic_optim, tb):
    n_samples = 0

    for global_step in itertools.count():
        if n_samples >= config.max_steps:
            break
        #  if global_step % config.save_every == 0:
       #      print(f"Saved at {global_step}. Progress:{n_samples / config.max_steps:.2f}")
       #      tb.add_object("model", agent, global_step)
        trajectory, env_return = gather_trajectory(env, agent)

        critic_info = svg.critic(trajectory, agent, critic_optim,batch_size=config.batch_size)
        # ascend the gradient
        actor_info= svg.actor(trajectory, agent, env, actor_optim,batch_size=config.batch_size)
        tb.add_scalar("train/return",env_return,n_samples)
        scalars_to_tb(tb,{**actor_info,**critic_info},n_samples)
        n_samples += 200
        print(global_step, env_return)

        if global_step % config.save_every == 0 and global_step > 0:
            info = eval_policy(PendulumEnv(), agent, log_dir=tb.objects_path)
            print(info)

if __name__ == "__main__":
    main()
