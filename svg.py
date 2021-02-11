from typing import NamedTuple

import jax
from jax import numpy as jnp

import utils
from agent import Actor
from config import config


class ModelTransition(NamedTuple):
    action: int = 0
    state: int = 0


class ReverseAgent:
    def __init__(self, dynamics):
        self._agent = Actor(action_space=dynamics.action_space)
        self._dynamics = dynamics
        self._key_gen = None
        self.control_fn = jax.jit(self.control_fn)
        self.training_step = self.training_step

    def initial_params(self, key):
        self._grad_fn = jax.value_and_grad(self.vg_loss)
        x = self._dynamics.get_state()
        params = self._agent.init(key, x)
        self.master_key = key
        return params

    def act(self, params, state):
        self.master_key, key = jax.random.split(self.master_key)
        action_noise = jax.random.normal(self.master_key)
        action = self.control_fn(params, state, action_noise)
        return action, {"action_noise": action_noise}

    def control_fn(self, params, state, action_noise):
        pi = self._agent.apply(params, state)
        action = pi.loc + pi.scale * action_noise
        return action

    def post_action(self,  *args):
        return

    def training_step(self, params, model_params, transitions, scale):
        loss, grad = self._grad_fn(params, model_params, transitions)
        grad = utils.clip_gradients(grad, config.grad_clip)
        grad_norm = utils.global_norm(grad)
        return grad, {"loss": loss, "grad_norm": grad_norm}

    def _vg_loss(self, state, transition, params, model_params):
        t, state = state
        pi = self._agent.apply(params, state)
        action = pi.loc + pi.scale * transition.action_noise
        model_noise = self._dynamics.infer_noise(model_params, state, action, transition.next_state)
        next_state, reward = self._dynamics(model_params, state, action, model_noise)
        state = next_state
        return (t + 1, state), reward

    def vg_loss(self, params, model_params, transitions):
        partial_ = jax.partial(self._vg_loss, params=params, model_params=model_params)
        state = transitions[0].state
        xs = list(zip(*transitions))
        xs = utils.Transition(*[jnp.stack(x) for x in xs])
        _, rewards = jax.lax.scan(partial_, init=(0, state), xs=xs)
        return - jnp.sum(rewards)

    def zero_grad(self):
        self._t = 0
        self._returns = 0


def reinforce(model, samples):
    reinforce_loss = 0.
    for transition, value in zip(*samples):
        states, actions, _, _, _ = transition
        policy = model(states)
        loss = -utils.log_prob(actions, policy).sum(axis=-1) * value
        loss = utils.scale_gradients(loss, scale=(1 / len(samples[0])))
        loss = jnp.mean(loss)
        reinforce_loss += loss
    return reinforce_loss
