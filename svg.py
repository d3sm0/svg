import operator

from models import Policy

from typing import NamedTuple
import jax

import jax.numpy as jnp


class Transition(NamedTuple):
    state: jnp.array
    action: jnp.array
    reward: jnp.array
    next_state: jnp.array
    done: jnp.array


class Trajectory:
    def __init__(self):
        self._data = []

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __repr__(self):
        return f"N:{self.__len__()}"

    def append(self, transition):
        self._data.append(transition)


@jax.jit
def get_grad_norm(grad):
    grad_norm = 0
    for x in jax.tree_leaves(grad):
        grad_norm += jnp.linalg.norm(x)
    return grad_norm


def recreate_transition(state, transition, dynamics, policy):
    a_hat, scale = Policy(1).apply({"params": policy}, state)
    eps = (transition.action - a_hat) / scale
    action = a_hat + scale * jax.lax.stop_gradient(eps)

    s1_hat, sigma = dynamics(state, action)
    eta = (transition.next_state - s1_hat) / sigma
    next_state = s1_hat + sigma * jax.lax.stop_gradient(eta)

    return action, next_state, (scale, sigma)


def _one_step(state, transition, params, dynamics):
    t, state = state
    action, next_state, noise = recreate_transition(state, transition, dynamics, params)
    reward = dynamics.reward(state, action)
    state = next_state
    return (t + 1, state), reward


def unroll(params, dynamics, transitions):
    partial_ = jax.partial(_one_step, params=params, dynamics=dynamics)
    state = transitions[0].state
    xs = list(zip(*transitions))
    xs = Transition(*[jnp.stack(x) for x in xs])
    _, rewards = jax.lax.scan(partial_, init=(0, state), xs=xs)
    return - jnp.sum(rewards)


@jax.partial(jax.jit, static_argnums=(0, 3))
def train(dynamics, policy, pi_optim, traj):
    total_return, grad = jax.value_and_grad(unroll)(policy, dynamics, traj)
    true_norm = get_grad_norm(grad)
    pi_optim = pi_optim.apply_gradient(grad)
    return pi_optim, {"agent/return": total_return, "agent/grad_norm": true_norm}


@jax.jit
def control_fn(key_gen, params, state):
    eps = jax.random.normal(key_gen, shape=(1, 1))
    mu, std = Policy(1).apply({"params": params}, state)
    action = mu + std * eps
    return action.squeeze(1)


def generate_episode(env, key_gen, params):
    state, _, done, info = env.reset()
    trajectory = Trajectory()
    while not done:
        action = control_fn(next(key_gen), params, state)
        next_state, reward, done, info = env.step(action)
        trajectory.append(Transition(state, action, reward, next_state, done))
        state = next_state
    return trajectory, info
