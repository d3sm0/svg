import copy
from collections import Iterable

import torch
from torch import nn as nn
from torch.nn import functional as F


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.orthogonal_(m.weight)
        torch.nn.init.zeros_(m.bias)


def polyak_update(params, target_params, tau=1.):
    with torch.no_grad():
        # zip does not raise an exception if length of parameters does not match.
        for param, target_param in zip(params, target_params):
            target_param.data.mul_(1 - tau)
            torch.add(target_param.data, param.data, alpha=tau, out=target_param.data)


class Agent(nn.Module):
    def __init__(self, obs_dim, action_dim, h_dim=100):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.actor = nn.Sequential(nn.Linear(obs_dim, h_dim), nn.SELU(),
                                   # nn.Linear(h_dim, h_dim), nn.SELU(),
                                   # nn.Linear(h_dim, h_dim), nn.SELU(),
                                   nn.Linear(h_dim, 2 * action_dim))

        self.critic = nn.Sequential(
            nn.Linear(obs_dim + action_dim, h_dim),
            nn.SELU(),
            nn.Linear(h_dim, h_dim),
            nn.SELU(),
            nn.Linear(h_dim, h_dim),
            nn.SELU(),
            nn.Linear(h_dim, 1),
        )

        self.target_critic = copy.deepcopy(self.critic)

    def forward(self, s):
        out = self.actor(s)
        mu, sigma = torch.split(out, self.action_dim, -1)
        sigma = F.softplus(sigma)
        return mu, sigma

    def value(self, state, action):
        state_action = torch.cat([state, action], -1)
        return self.critic(state_action)

    def target_value(self, state, action):
        state_action = torch.cat([state, action], -1)
        return self.target_critic(state_action)

    def get_action(self, s):
        with torch.no_grad():
            mu, sigma = self.forward(s)
        eps = torch.randn(size=mu.shape)
        return mu + sigma * eps, eps

    def rsample(self, s):
        mu, sigma = self.forward(s)
        eps = torch.randn(size=mu.shape).detach()
        return mu + sigma * eps

    def update_target(self, tau):
        polyak_update(self.critic.parameters(), self.target_critic.parameters(), tau)
