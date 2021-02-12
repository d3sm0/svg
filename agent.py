from config import config
from flax import linen
from jax import numpy as jnp

from utils import Density


class Actor(linen.Module):
    """Class defining the actor-critic model."""

    action_space: int
    std = config.agent_std
    features: int = 32  # 32

    def setup(self):
        self.fc = linen.Dense(features=self.features)
        self.out = linen.Dense(features=2 * self.action_space)
        self.b1 = ResNetBlock()

    def __call__(self, x) -> Density:
        h = self.fc(x)
        h = linen.tanh(h)
        h = self.b1(h)
        out = self.out(h)
        mu, log_sigma = jnp.split(out, 2, -1)
        sigma = linen.softplus(log_sigma) + 1e-3
        return Density(mu, sigma)


class ResNetBlock(linen.Module):
    """ResNet block."""
    features = 32

    def setup(self):
        self.fc = linen.Dense(features=self.features)
        self.fc1 = linen.Dense(features=self.features)

    def __call__(self, x):
        residual = x
        y = self.fc(x)
        y = linen.tanh(y)
        y = self.fc1(y)
        return linen.tanh(residual + y)


class Dynamics(linen.Module):
    state_dim: int
    h_dim: int = 32
    std: float = config.learned_model_std

    def setup(self):
        self.fc = linen.Dense(self.h_dim)
        self.b = ResNetBlock()
        self.out = linen.Dense(2 * self.state_dim)

    def __call__(self, state, action):
        x = jnp.concatenate((state, action), axis=-1)
        h = self.fc(x)
        h = linen.tanh(h)
        h = self.b(h)
        out = self.out(h)
        mu, _ = jnp.split(out, 2, -1)
        sigma = jnp.ones_like(mu) * self.std
        return Density(mu, sigma)
