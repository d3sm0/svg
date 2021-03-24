import gym
from gym.envs.classic_control.pendulum import PendulumEnv
import torch
import math
import numpy as np

_env = PendulumEnv()
G, L, M, DT = _env.g, _env.l, _env.m, _env.dt
_MAX_TORQUE = _env.max_torque
_MAX_SPEED = _env.max_speed
del _env


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class DifferentiablePendulum(gym.envs.classic_control.PendulumEnv):
    def __init__(self):
        super(DifferentiablePendulum, self).__init__()

        def _reward(states, actions):
            theta_cos, theta_sin, thetadot = states.unbind(dim=-1)
            theta = torch.atan2(theta_sin, theta_cos)
            clamped_actions = actions.clamp(-_MAX_TORQUE, _MAX_TORQUE)

            costs = angle_normalize(theta).pow(2) + 0.1 * thetadot.pow(2) + 0.001 * (clamped_actions.pow(2))
            rewards = -costs
            return rewards

        def _dynamics_in_stspace(th, thdot, action):
            u = torch.clamp(action, -_MAX_TORQUE, _MAX_TORQUE)

            newthdot = thdot + (-3 * G / (2 * L) * torch.sin(th + math.pi) + 3. / (M * L ** 2) * u) * DT
            newth = th + newthdot * DT
            newthdot = torch.clamp(newthdot, -_MAX_SPEED, _MAX_SPEED)
            return newth, newthdot

        self._dynamics_in_stspace = _dynamics_in_stspace

        def _dynamics(state, action):
            theta_cos, theta_sin, thdot = state.unbind(dim=-1)
            th = torch.atan2(theta_sin, theta_cos)

            newth, newthdot = _dynamics_in_stspace(th, thdot, action)
            next_state = torch.cat([torch.cos(newth), torch.sin(newth), newthdot])
            return next_state

        self.dynamics = _dynamics
        self.reward = _reward

    def reset(self):
        self.last_u = None
        self.state = np.array([0., 0.], dtype=np.float32)
        return self._get_obs()


if __name__ == "__main__":
    env1 = DifferentiablePendulum()
    env2 = DifferentiablePendulum()

    obs1 = (env1.reset(), 0)
    env2.state = torch.FloatTensor(env1.state)
    obs2 = (torch.FloatTensor(obs1[0]), 0)
    print(env1.state, env2.state)

    import numpy as np

    a = np.array([0.1])

    for i in range(10):
        state, reward = env2.dynamics(obs2[0], torch.FloatTensor(a)), env2.reward(obs2[0], torch.FloatTensor(a))
        obs1 = env1.step(a)
        obs2 = env2.step(torch.FloatTensor(a))
        print(np.linalg.norm(obs1[0] - obs2[0].numpy()), np.linalg.norm(obs1[0] - state.numpy()),
              np.linalg.norm(obs1[1] - obs2[1].numpy()), np.linalg.norm(obs1[1] - reward.numpy()))
