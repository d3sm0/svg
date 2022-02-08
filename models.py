import rlego
import torch
from torch import nn as nn
from torch._vmap_internals import vmap


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, h_dim=100):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(obs_dim, obs_dim), nn.SiLU(), rlego.GaussianPolicy(obs_dim, action_dim))

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


class ValueZero(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(ValueZero, self).__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, 6), nn.SiLU(), nn.Linear(6, 6), nn.SiLU())
        self.actor = Actor(6, action_dim)
        self.planner = Actor(6, action_dim)
        self.planner.load_state_dict(self.actor.state_dict())
        self.dynamics = Dynamics(6, action_dim)
        self.critic = Critic(6)

    def __call__(self, state):
        return self.actor(self.body(state))


class Dynamics(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Dynamics, self).__init__()
        self.dynamics = nn.Sequential(nn.Linear(obs_dim + action_dim, 6), nn.SiLU(), nn.Linear(6, obs_dim))
        self.reward = nn.Sequential(nn.Linear(obs_dim + action_dim + obs_dim, 6), nn.SiLU(), nn.Linear(6, 1))

    def __call__(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        s_tp1 = self.dynamics(sa)
        sas = torch.cat([state, action, s_tp1], dim=-1)
        r = self.reward(sas).squeeze(dim=-1)
        return s_tp1, r


class DynamicsLQR:
    def __init__(self, env):
        self.env = env

    def __call__(self, state, action):
        s_tp1 = vmap(self.env.dynamics_fn)(state, action)
        r = vmap(self.env.reward_fn)(state, action)
        return s_tp1, r
