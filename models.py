import rlego
import torch
from torch import nn as nn


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, h_dim=100):
        super().__init__()
        self.pi = rlego.GaussianPolicy(obs_dim, action_dim)

    def forward(self, s):
        out = self.pi(s)
        return out


class Critic(nn.Module):
    def __init__(self, obs_dim, h_dim=100):
        super().__init__()
        self.critic = nn.Sequential(nn.Linear(obs_dim, 6), nn.SiLU(), nn.Linear(6, 1))

    def forward(self, state):
        return self.critic(state)


class QFunction(nn.Module):
    def __init__(self, obs_dim, action_dim, h_dim=100):
        super().__init__()
        self.q = nn.Sequential(nn.Linear(obs_dim + action_dim, 6), nn.SiLU(), nn.Linear(6, 1))
        # self._critic = nn.Sequential(nn.Linear(obs_dim, ))
        # self._add_action = nn.Sequential(nn.Linear(h_dim + action_dim, h_dim), nn.ReLU())
        # self._out = nn.Linear(h_dim, 1)

    def forward(self, state, action):
        return self.q(torch.cat([state, action], dim=-1))
        # h = self._critic(state)
        # state_and_action = torch.cat([h, action], dim=-1)
        # h = self._add_action(state_and_action)
        # return self._out(h)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, h_dim=100):
        super(ActorCritic, self).__init__()
        self.actor = Actor(obs_dim, action_dim, h_dim)
        self.critic = Critic(obs_dim, h_dim)


class ActorValue(nn.Module):
    def __init__(self, obs_dim, action_dim, h_dim=100):
        super(ActorValue, self).__init__()
        self.actor = Actor(obs_dim, action_dim, h_dim)
        self.critic = QFunction(obs_dim, action_dim, h_dim)
        self.baseline = Critic(obs_dim)
        self.target_critic = Critic(obs_dim)
        self.target_critic.load_state_dict(self.baseline.state_dict())
