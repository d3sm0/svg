import gym
import torch
from gym.envs.classic_control.pendulum import PendulumEnv
import numpy as np
import math


_env = PendulumEnv()
G, L, M, DT = _env.g, _env.l, _env.m, _env.dt
_MAX_TORQUE = _env.max_torque
_MAX_SPEED = _env.max_speed
del _env


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class PendulumOriginal(gym.Env):
    """
    Observation space:
        0 cos(theta),
        1 sin(theta)
        2 thetadot        [-8, +8]
    """
    def __init__(self):
        super().__init__()
        self.env = PendulumEnv()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.reward_range = self.env.reward_range
        self.spec = self.env.spec
        self.metadata = self.env.metadata

        def _reward(states, actions):
            theta_cos, theta_sin, thetadot = states.unbind(dim=-1)
            theta = torch.atan2(theta_sin, theta_cos)

            clamped_actions = actions.clamp(-_MAX_TORQUE, _MAX_TORQUE)

            costs = angle_normalize(theta).pow(2) + 0.1 * thetadot.pow(2) + 0.001 * (clamped_actions.pow(2))
            rewards = -costs
            return rewards

        def _dynamics_in_stspace(th, thdot, action):
            u = torch.clamp(action, -_MAX_TORQUE, _MAX_TORQUE)

            newthdot = thdot + (-3*G/(2*L) * torch.sin(th + math.pi) + 3./(M*L**2)*u) * DT
            newth = th + newthdot*DT
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

    def render(self, mode='human'):
        return self.env.render(mode)

    def reset(self):
        high = torch.FloatTensor([np.pi, 1])
        self.state = torch.FloatTensor(self.env.np_random.uniform(low=-high, high=high))
        theta, thetadot = self.state
        obs = torch.stack([torch.cos(theta), torch.sin(theta), thetadot])
        self.last_u = None
        return obs

    @torch.no_grad()
    def step(self, action):
        th, thdot = self.state  # th := theta
        newth, newthdot = self._dynamics_in_stspace(th, thdot, action)
        self.state = torch.cat([newth, newthdot])

        next_state = torch.cat([torch.cos(newth), torch.sin(newth), newthdot])

        # Compute reward
        clamped_actions = action.clamp(-_MAX_TORQUE, _MAX_TORQUE)
        costs = angle_normalize(th).pow(2) + 0.1 * thdot.pow(2) + 0.001 * (clamped_actions.pow(2))
        reward = -costs
        return next_state, reward, None, {}

    def seed(self, seed=None):
        return self.env.seed(seed)

    def close(self):
        return self.env.close()
