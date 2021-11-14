from typing import NamedTuple

import gym
import numpy as np
import torch

from . import ENV_IDS


class EnvSpec(NamedTuple):
    observation_space: int
    action_space: int


class ActionBound(NamedTuple):
    low: float
    high: float


def _get_action_bound(bound: gym.spaces.Box):
    assert np.isfinite(bound.low).all() and np.isfinite(bound.high).all()
    # torch.clamp does not seem to support vector
    return ActionBound(bound.low.max(), bound.high.max())


class Wrapper(gym.Wrapper):
    def __init__(self, env, horizon, gamma=0.99):
        super(Wrapper, self).__init__(env)
        assert hasattr(env.unwrapped, "reward"), "Env must expose reward_fn for SVG"
        self.reward = env.unwrapped.reward
        self.env_spec = EnvSpec(self.env.observation_space.shape[0], self.env.action_space.shape[0])
        self._action_bound = _get_action_bound(self.env.action_space)
        self.horizon = horizon
        if hasattr(env, "gamma"):
            gamma = gamma
        self.gamma = gamma
        self.returns = None
        self.t = None

    def step(self, action):
        assert torch.isfinite(action), f"action is {action}"
        action = torch.clamp(action, self._action_bound.low, self._action_bound.high)
        action = action.numpy()
        next_state, r, d, _ = super(Wrapper, self).step(action)
        info = {
            "env/reward": r,
            "env/avg_reward": self.returns / (self.t + 1),
            "env/returns": self.returns,
            "env/steps": self.t
        }
        if self.t >= self.horizon or d is True:
            d = True
        self.t += 1
        self.returns += r
        return *self._to_torch(next_state, r, d), info

    @staticmethod
    def _to_torch(s, r, d):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, dtype=torch.float32)
        if not isinstance(r, torch.Tensor):
            r = torch.tensor(r, dtype=torch.float32)
        d = torch.tensor(d, dtype=torch.float32)
        return s, r, d

    def reset(self, **kwargs):
        state = super(Wrapper, self).reset()
        self.returns = 0
        self.t = 0
        return *self._to_torch(state, 0, False), {}


def make_env(env_id="lqg", horizon=200):
    assert env_id in ENV_IDS, f"env_id:{env_id} not in  {ENV_IDS}."
    env = gym.make(env_id)
    env = Wrapper(env, horizon=horizon)
    return env
