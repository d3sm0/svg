import operator
from functools import reduce
from typing import Dict, Text, Any, NamedTuple

import gym
import jax
from flax.core import FrozenDict
from jax import numpy as jnp


class Trajectory:
    def __init__(self):
        self._data = []
        self._n_samples = 0

    def __len__(self):
        return len(self._data)

    def __reversed__(self):
        return reversed(self._data)

    def get_first_state(self):
        return self._data[0][0]

    @property
    def n_samples(self):
        return self._n_samples

    def append(self, step):
        self._data.append(step)
        self._n_samples += 1


class Transition(NamedTuple):
    state: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    next_state: jnp.ndarray
    action_noise: jnp.zeros = 0.
    model_noise: jnp.zeros = 0.


class Density(NamedTuple):
    loc: jnp.zeros = 0.
    scale: jnp.zeros = 0.


@jax.jit
def log_prob(value: jnp.ndarray, density: Density) -> jnp.ndarray:
    # compute the variance
    var = (density.scale ** 2)
    log_scale = jax.lax.log(density.scale)
    return - ((value - density.loc) ** 2) / (2 * var) - log_scale - 0.5 * jax.lax.log(2 * jnp.pi)


@jax.jit
def kl_normal_normal(p: Density, q: Density) -> jnp.ndarray:
    var_ratio = (p.scale / q.scale) ** 2
    t1 = ((p.loc - q.loc) / q.scale) ** 2
    return jnp.sum(0.5 * (var_ratio + t1 - 1 - jnp.log(var_ratio)))


@jax.partial(jax.jit, static_argnums=(1,))
def clip_gradients(grads: FrozenDict, value: int = 1.) -> FrozenDict:
    return jax.tree_map(lambda t: jnp.clip(t, -value, value), grads)


@jax.jit
def global_norm(grad):
    return jnp.sqrt(sum([jnp.sum(jnp.square(x)) for x in jax.tree_leaves(grad)]))


@jax.jit
def normalize_gradients(grads, max_norm):
    g_norm = global_norm(grads)
    g_norm = jnp.minimum(g_norm, max_norm)
    grads = jax.tree_map(lambda t: (t / g_norm) * max_norm, grads)
    return grads, g_norm


@jax.jit
def scale_gradients(tensor: jnp.ndarray, scale: float) -> jnp.ndarray:
    tensor = scale * tensor + (1 - scale) * jax.lax.stop_gradient(tensor)
    return tensor


def eval_policy(env: gym.Env, control_fn, actor_params) -> Dict[Text, Any]:
    state, _, done, _ = env.reset()
    info = {}
    while not done:
        action = control_fn(actor_params, state, 0.)
        state, reward, done, info = env.step(action)
    return info


@jax.jit
def entropy(param: Density) -> jnp.array:
    return 0.5 + 0.5 * jnp.log(2 * jnp.pi) + jnp.log(param.scale)


@jax.jit
def l1_penalty(params: FrozenDict) -> jnp.array:
    weight_penalty_params = jax.tree_leaves(params)
    w_l1 = sum([jnp.sum(jnp.square(x)) for x in weight_penalty_params if x.ndim > 1])
    return w_l1


@jax.partial(jax.jit, static_argnums=(2,))
def polyak_average(new_params: FrozenDict, old_params: FrozenDict, beta: float) -> FrozenDict:
    _avg = lambda x, y: beta * x + (1 - beta) * y
    actor_params = jax.tree_multimap(lambda x, y,: _avg(x, y), new_params, old_params)
    return actor_params


def prod(iterable):
    return reduce(operator.mul, iterable, 1)
