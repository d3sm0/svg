import torch
from torch import nn as nn
from torch.nn import functional as F


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.orthogonal_(m.weight)
        torch.nn.init.zeros_(m.bias)


class Agent(nn.Module):
    def __init__(self, obs_dim, action_dim, h_dim=100):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.actor = nn.Sequential(nn.Linear(obs_dim, h_dim),
                                   nn.SELU(),
                                   # nn.Linear(h_dim, h_dim),
                                   # nn.SELU(),
                                   nn.Linear(h_dim, h_dim),
                                   nn.SELU(),
                                   nn.Linear(h_dim, 2 * action_dim)
                                   )

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, h_dim),
            nn.SELU(),
            # nn.Linear(h_dim, h_dim),
            # nn.SELU(),
            nn.Linear(h_dim, h_dim),
            nn.SELU(),
            nn.Linear(h_dim, 1),
        )

    def forward(self, s):
        out = self.actor(s)
        mu, sigma = torch.split(out, self.action_dim, -1)
        sigma = F.softplus(sigma)
        return mu, sigma

    def value(self, x):
        return self.critic(x)

    def get_action(self, s):
        with torch.no_grad():
            mu, sigma = self.forward(s)
        eps = torch.randn(size=mu.shape)
        return mu + sigma * eps, eps
