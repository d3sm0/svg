import copy

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


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, h_dim=100):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(obs_dim, h_dim), nn.SELU(), nn.Linear(h_dim, h_dim), nn.SELU(),
                                   nn.Linear(h_dim, h_dim), nn.SELU(), nn.Linear(h_dim, 2 * action_dim))

    def forward(self, s):
        out = self.actor(s)
        mu, sigma = torch.split(out, out.shape[-1] // 2, -1)
        sigma = F.softplus(sigma)
        return mu, sigma


class Critic(nn.Module):
    def __init__(self, obs_dim, h_dim=100):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, h_dim),
            nn.SELU(),
            # nn.Linear(h_dim, h_dim),
            # nn.SELU(),
            nn.Linear(h_dim, h_dim),
            nn.SELU(),
            nn.Linear(h_dim, 1),
        )


class QFunction(nn.Module):
    def __init__(self, obs_dim, action_dim, h_dim=100):
        super().__init__()

        self._critic = nn.Sequential(
            nn.Linear(obs_dim + action_dim, h_dim),
            nn.SELU(),
            # nn.Linear(h_dim, h_dim),
            # nn.SELU(),
            nn.Linear(h_dim, h_dim),
            nn.SELU(),
            nn.Linear(h_dim, 1),
        )

    def forward(self, state_and_action):
        return self._critic(state_and_action)


class ActorCritc(nn.Module):
    def __init__(self, obs_dim, action_dim, h_dim=100):
        super(ActorCritc, self).__init__()
        self.actor = Actor(obs_dim, action_dim, h_dim)
        self.critic = Critic(obs_dim, h_dim)


class ActorValue(nn.Module):
    def __init__(self, obs_dim, action_dim, h_dim=100):
        super(ActorValue, self).__init__()
        self.actor = Actor(obs_dim, action_dim, h_dim)
        self.critic = QFunction(obs_dim, action_dim, h_dim)
        self.target_critic = copy.deepcopy(self.critic)
