import gym
import jax
from jax import numpy as jnp


class Lqg(gym.core.Env):
    action_dim = 1
    state_dim = 2

    def __init__(self):
        action_dim = 1
        state_dim = 2
        self.action_space = gym.spaces.Box(-1,1, shape=(action_dim,))
        self.observation_space = gym.spaces.Box(-1, 1, shape=(state_dim,))

        self.initial_state = jnp.zeros(state_dim)

        # Dynamics
        self.A = jnp.array([[1., 0.], [0., 1.]])
        self.B = jnp.array([[0.3],
                            [-0.3]])
        self.AB = jnp.hstack((self.A, self.B))

        # Cost
        self.Q = 0.1 * jnp.eye(state_dim)
        self.R = jnp.array([[1]])

        def _reward(state, action):
            cost = (state.transpose() @ self.Q @ state + action.transpose() @ self.R @ action)
            return -cost

        def _dynamics(state, action):
            return self.AB @ jnp.concatenate((state, action))

        self.f = jax.jit(_dynamics)
        self.r = jax.jit(_reward)

    def reset(self):
        self.state = self.initial_state
        return self.state

    def render(self, mode='human'):
        print(self.state)

    def step(self, action):
        r = self.r(self.state, action)
        #action = jax.lax.clamp(-2.,action,  2.)
        s1 = self.f(self.state, action)
        self.state = s1
        return s1, r, False, {}
