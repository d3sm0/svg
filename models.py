import warnings
import functools
import numpy as np
from flax import linen as nn
from flax import optim
import jax
import jax.numpy as jnp


class Policy(nn.Module):
    num_outputs: int

    @nn.compact
    def __call__(self, x):
        dtype = jnp.float32
        x = nn.Dense(features=3, name='hidden', dtype=dtype)(x)
        x = nn.relu(x)
        mu = nn.Dense(features=self.num_outputs, name='pi', dtype=dtype)(x)
        return mu, jnp.ones_like(mu) * 0.1


# @functools.partial(jax.jit, static_argnums=1)
def get_initial_params(key: np.ndarray):
    init_shape = jnp.ones((3,), jnp.float32)
    initial_params = Policy(1).init(key, init_shape)['params']
    return initial_params


class RealDynamics:
    def __init__(self, env):
        self._f = env.dynamics
        warnings.warn("Bound constraints in step(). Strange things might happen.")
        self.reward = env.reward
        self.std = 0.01

    def __call__(self, s, a):
        mu, sigma = self._f(s, a), self.std
        return mu, sigma
