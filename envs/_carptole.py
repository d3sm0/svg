import gym
import jax
from gym.envs import classic_control
from jax import numpy as jnp

from envs.pendulum import angle_normalize


class Cartpole(classic_control.CartPoleEnv):
    def __init__(self):
        super(Cartpole, self).__init__()
        self.action_space = gym.spaces.Box(-1, 1, shape=(1,))

        def _reward(state, action):
            (x, x_dot, theta, theta_dot) = state
            cost = angle_normalize(theta) ** 2
            cost += 0.1 * theta_dot ** 2 + 0.1 * (x - self.x_threshold) ** 2
            cost += .001 * (action ** 2)
            return jnp.sum(-cost)

        def _dynamics(state: jnp.array, action: jnp.array) -> jnp.array:
            (x, x_dot, theta, theta_dot) = state
            force = self.force_mag * action.squeeze()
            costheta = jnp.cos(theta)
            sintheta = jnp.sin(theta)
            # For the interested reader:
            # https://coneural.org/florian/papers/05_cart_pole.pdf
            temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
            thetaacc = (self.gravity * sintheta - costheta * temp) / (
                    self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
            xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
            new_x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            new_th = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
            return jnp.array([new_x, x_dot, new_th, theta_dot])

        self.r = jax.jit(_reward)
        self.f = jax.jit(_dynamics)
        self.reset()

    def step(self, action):
        assert not jnp.isnan(action)
        # if self.done:
        #   self.state = self.reset()
        #   return self.state, 0., False, {}
        action = jnp.squeeze(action)
        reward = self.r(self.state, action)
        # action = jax.lax.clamp(-2., action, 2.)
        # self.done = self._is_terminal(self.state[2], self.state[0])
        # reward = (1. - self.done)
        next_state = self.f(self.state, action)
        # [x, x_dot, theta, theta_dot] = next_state

        # next_state = jnp.clip(next_state, -10., 10.)
        # x = jnp.clip(x, -10.,10.)
        # x_dot = jnp.clip(x_dot, -4., 4.)
        # theta = jnp.clip(theta, -1.2, 1.2)
        # theta_dot = jnp.clip(theta_dot, -2., 2.)

        # next_state = jnp.array([x, x_dot, theta, theta_dot])

        self.state = next_state
        return next_state, reward, self.done, {}

    def _is_terminal(self, theta, x):
        done = (
                x < -self.x_threshold
                or x > self.x_threshold
                or theta < -self.theta_threshold_radians
                or theta > self.theta_threshold_radians
        )
        return bool(done)

    def reset(self):
        self.done = False

        #self.state = jnp.array([10., 0., -1.2, 0.])
        # next_state = jnp.clip(next_state, -10., 10.)
        # x = jnp.clip(x, -10.,10.)
        # x_dot = jnp.clip(x_dot, -4., 4.)
        # theta = jnp.clip(theta, -1.2, 1.2)
        # theta_dot = jnp.clip(theta_dot, -2., 2.)

        self.state = jnp.zeros(shape=(4,))
        return self.state
