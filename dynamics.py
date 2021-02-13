from collections import Callable

#import jax
#from jax import numpy as jnp
#
#from agent import Dynamics
#from config import config


class Model:
    f: Callable
    r: Callable
    observation_space: int
    action_space: int

    def __init__(self, r, env_spec):
        self.r = jax.jit(r)

        self.observation_space = env_spec.observation_space
        self.action_space = env_spec.action_space
        self.training_step = jax.jit(self.training_step)
        self._dynamics = Dynamics(self.observation_space)

        def f(model_params, state, action, noise):
            delta = self._dynamics.apply(model_params, state, action)
            next_state = state + delta.loc + noise * config.learned_model_std
            return next_state

        self.f = jax.jit(f)

        f_s = jax.jacfwd(f, argnums=1)
        f_a = jax.jacfwd(f, argnums=2)
        self.f_s = jax.jit(f_s)  # jax.partial(normalize, f_s))
        self.f_a = jax.jit(f_a)  # jax.partial(normalize, f_a))
        self.r_s = jax.jit(jax.grad(self.r, argnums=0))
        self.r_a = jax.jit(jax.grad(self.r, argnums=1))

    def __call__(self, params, state, action, noise):
        next_state = self.f(params, state, action, noise)
        reward = self.r(state, action)
        return next_state, reward

    def get_state(self):
        return jnp.zeros(shape=self.observation_space)

    def training_step(self, model_params, batch):
        state, action, rewards, next_state = batch

        def loss_fn(model_params):
            delta = self._dynamics.apply(model_params, state, action)
            delta_true = next_state - state
            loss = 0.5 * ((delta_true - delta.loc) ** 2).sum(axis=1)

            # pred_reward = ((total_reward - v_reward(state + delta.loc, action)) ** 2).mean()
            return loss.mean()  # + config.l1_penalty * utils.l1_penalty(model_params)#

        value, grad = jax.value_and_grad(loss_fn)(model_params)

        return value, grad

    def infer_noise(self, params, state, action, next_state):
        delta = self._dynamics.apply(params, state, action)
        noise = (next_state - (state + delta.loc)) / config.learned_model_std
        return jax.lax.stop_gradient(noise)

    def initial_params(self, key):
        params = self._dynamics.init(key, jnp.zeros(shape=(1, self.observation_space)),
                                     jnp.zeros(shape=(1, self.action_space)))
        return params


class FakeModel:
    def __init__(self, f, r, env_spec):
        self.observation_space = env_spec.observation_space
        self.action_space = env_spec.action_space

        self.mu = jax.jit(f)

        def _f(params, state, action, noise):
            return f(state, action) + noise

        self.f = jax.jit(_f)
        self.r = jax.jit(r)

        self.f_s = jax.jit(jax.jacfwd(self.f, argnums=1))
        self.f_a = jax.jit(jax.jacfwd(self.f, argnums=2))
        self.r_s = jax.jit(jax.grad(self.r, argnums=0))
        self.r_a = jax.jit(jax.grad(self.r, argnums=1))
        self.infer_noise = jax.jit(self.infer_noise)

    def __call__(self, params, state, action, noise):
        next_state = self.f(params, state, action, noise)
        reward = self.r(state, action)
        return next_state, reward

    def get_state(self):
        return jnp.zeros(shape=self.observation_space)

    def infer_noise(self, params, state, action, next_state):
        error = (next_state - self.mu(state, action)) / config.learned_model_std
        return jax.lax.stop_gradient(error)

    def training_step(self, *args):
        return {}

    def initial_params(self, key):
        return {}


class Buffer:
    def __init__(self):
        self._data = []
        self._n_trajectories = 0
        self._n_samples = 0

    def append(self, trajectory):
        if self._n_trajectories > config.buffer_size:
            self._n_trajectories -= 1
            self._data.pop(0)
        self._data.append(trajectory)
        self._n_trajectories += 1
        self._n_samples += trajectory.n_samples

    def sample(self, key, batch_size):
        key1, key2 = jax.random.split(key)
        trajectory_idx = jax.random.randint(key1, (1,), 0, self._n_trajectories).item()

        sample_idx = jax.random.randint(key2, (batch_size,), 1, self._data[trajectory_idx].n_samples - 1)

        states = []
        actions = []
        rewards = []
        next_state = []

        for idx in sample_idx:
            trajectory = self._data[trajectory_idx]
            transition = trajectory[idx]
            states.append(transition.state)
            next_state.append(transition.next_state)
            rewards.append(transition.reward)
            actions.append(transition.action)

        return jnp.stack(states), jnp.stack(actions), jnp.stack(rewards), jnp.stack(next_state)
