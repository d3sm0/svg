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
    env = DifferentiablePendulum()

    s = env.reset()
    for i in range(10):
        a = env.action_space.sample()
        s1, r, d, info = env.step(a)
        _s1 = env.dynamics(torch.from_numpy(s).float(), torch.from_numpy(a).float())
        assert np.allclose(_s1, s1), print(_s1, s1)
        s = s1
