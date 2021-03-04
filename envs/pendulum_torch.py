import math

import gym
import numpy as np
import torch
from gym.envs import classic_control


def angle_normalize(x):
    return ((x + math.pi) % (2 * math.pi)) - math.pi


def _obs_to_th(obs):
    return obs
    # cos_th, sin_th, thdot = obs
    # th = torch.arctan(sin_th / cos_th)
    # return th.reshape((1,)), thdot.reshape((1,))


def _th_to_obs(th, thdot):
    # cos_th, sin_th = torch.cos(th), torch.sin(th)
    next_state = torch.tensor([th, thdot], dtype=torch.float32)
    return next_state


class Pendulum(classic_control.PendulumEnv):

    def __init__(self, horizon=200):
        super(Pendulum, self).__init__()
        high = np.array([1., self.max_speed])
        self.max_speed = 8.
        self.action_space = gym.spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-high, high=high, dtype=np.float32)
        self.viewer = None
        self.state = None
        self.horizon = horizon
        self.last_u = None
        self.reset()

        def _dynamics(state, action):
            th, thdot = _obs_to_th(state)
            newthdot = (thdot + (-3 * self.g / (2 * self.l) * torch.sin(th + math.pi) + 3.0 / (
                    self.m * self.l ** 2) * action) * self.dt)
            newth = th + newthdot * self.dt
            next_state = _th_to_obs(newth, newthdot)
            return next_state

        def _reward(state, action):
            th, th_dot = _obs_to_th(state)
            cost = torch.sum(angle_normalize(th) ** 2 + 0.1 * th_dot ** 2 + 0.001 * (action ** 2))
            return -cost

        self._f = _dynamics
        self._r = _reward

    def reset(self):
        self.state = torch.tensor((-1.,  0.))
        return self.state

    def step(self, action):
        self.last_u = action
        reward = self._r(self.state, action)
        next_state = self._f(self.state, action)
        th, th_dot = _obs_to_th(next_state)
        th_dot = torch.clamp(th_dot, -self.max_speed, self.max_speed)
        th = th + th_dot * self.dt
        next_state = _th_to_obs(th, th_dot)
        self.state = next_state
        return next_state, reward, False, {}
