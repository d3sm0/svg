import warnings

import torch
from torch import nn as nn
from torch.nn import functional as F


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.orthogonal_(m.weight)
        torch.nn.init.zeros_(m.bias)


class Policy(nn.Module):
    def __init__(self, obs_dim, action_dim, h_dim=32):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.fc = nn.Sequential(nn.Linear(obs_dim, h_dim),
                                nn.Tanh(),
                                nn.Linear(h_dim, h_dim),
                                nn.Tanh()
                                )
        self.out = nn.Linear(h_dim, 2 * action_dim)

        self._value = nn.Sequential(nn.Linear(obs_dim, h_dim),
                                    nn.Tanh(),
                                    nn.Linear(h_dim, h_dim),
                                    nn.Tanh(),
                                    nn.Linear(h_dim, 1),
                                    )
        self.apply(weights_init)

    def forward(self, s):
        h = self.fc(s)
        out = self.out(h)
        mu, sigma = torch.split(out, self.action_dim, -1)
        sigma = F.softplus(sigma).clamp_max(1.)
        return mu, sigma + 1e-3

    def value(self, x):
        return self._value(x)

    def sample(self, s):
        mu, sigma = self(s)
        eps = torch.randn(mu.shape)
        action = mu + sigma * eps
        return action.detach()


class RealDynamics(nn.Module):
    def __init__(self, env):
        super(RealDynamics, self).__init__()
        self._f = env.dynamics
        warnings.warn("Bound constraints in step(). Strange things might happen.")
        self.reward = env.reward
        self.std = 0.01

    def __call__(self, s, a):
        mu, sigma = self._f(s, a), self.std
        return mu, sigma


class Reward(nn.Module):
    def __init__(self, obs_dim, h_dim):
        super().__init__()
        self.out = nn.Sequential(nn.Linear(obs_dim, h_dim),
                                 nn.ReLU(),
                                 nn.Linear(h_dim, h_dim), nn.ReLU(),
                                 nn.Linear(h_dim, 1))

    def forward(self, s, a):
        return self.out(torch.cat((s, a), dim=-1)).view(-1)


class Dynamics(nn.Module):
    def __init__(self, env, h_dim=32, learn_reward=False, std=1.):
        super().__init__()
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self.reward = env.unwrapped.reward

        self.fc = nn.Sequential(
            nn.Linear(obs_dim + action_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, h_dim),
            nn.ELU()
        )
        self.out = nn.Linear(h_dim, 2 * obs_dim)
        self.std = std

        self.apply(weights_init)

    def forward(self, s, a):
        h = self.fc(torch.cat((s, a), dim=-1))
        dist = self.out(h)
        mu, sigma = torch.split(dist, s.shape[-1], -1)
        sigma = F.softplus(sigma) + 1e-3
        return nn.Sigmoid()(sigma), torch.clamp(sigma, -1., 1.)

    def __call__(self, s, a):
        s_tp1, sigma = self.forward(s, a)
        return s_tp1, sigma
