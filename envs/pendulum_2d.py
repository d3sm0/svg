import math

import gym
import numpy as np
import torch
from gym.envs.classic_control import pendulum


class Pendulum2D(pendulum.PendulumEnv):

    def __init__(self):
        super(Pendulum2D, self).__init__()
        high = np.array([1., self.max_speed])
        self.observation_space = gym.spaces.Box(low=-high, high=high, dtype=np.float32)
        self.state = None

        def _dynamics(state, action):
            th, thdot = state
            newthdot = (thdot + (-3 * self.g / (2 * self.l) * torch.sin(th + math.pi) + 3.0 / (
                    self.m * self.l ** 2) * action) * self.dt)
            newth = th + newthdot * self.dt
            newthdot = torch.clamp(newthdot, -self.unwrapped.max_speed, self.unwrapped.max_speed)
            next_state = torch.cat([newth, newthdot])
            return next_state

        def _reward(state, action):
            th, th_dot = state
            # cos_th = torch.cos(th)
            th = pendulum.angle_normalize(th)
            # print(th)
            cost = th ** 2 + 0.1 * th_dot ** 2 + 0.001 * (action ** 2)
            return -cost.sum()

        self.dynamics = _dynamics
        self.reward = _reward

    def reset(self):
        self.state = torch.tensor((1, 0.), dtype=torch.float32)
        self.last_u = None
        return self.state

    def step(self, action):
        # return super(Pendulum2D, self).step(action)
        self.last_u = action
        reward = self.reward(self.state, action)
        next_state = self.dynamics(self.state, action)
        self.state = next_state
        return self.state, reward, False, {}


class Pendulum3D(pendulum.PendulumEnv):

    def __init__(self):
        super(Pendulum3D, self).__init__()

        def _dynamics(state, action):
            th, thdot = state
            newthdot = (thdot + (-3 * self.g / (2 * self.l) * torch.sin(th + math.pi) + 3.0 / (
                    self.m * self.l ** 2) * action) * self.dt)
            newth = th + newthdot * self.dt
            newthdot = torch.clamp(newthdot, -self.unwrapped.max_speed, self.unwrapped.max_speed)
            next_state = torch.cat([newth, newthdot])
            return next_state

        def _reward(state, action):
            th, th_dot = state
            th = pendulum.angle_normalize(th)
            cost = th ** 2 + 0.1 * th_dot ** 2 + 0.001 * (action ** 2)
            return -cost.sum()

        self.dynamics = _dynamics
        self.reward = _reward

    def reset(self):
        self.state = torch.tensor((1, 0.), dtype=torch.float32)
        self.last_u = None
        return self._get_obs()
