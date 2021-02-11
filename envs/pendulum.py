import gym
import jax

from gym.envs import classic_control
from jax import numpy as jnp


@jax.jit
def angle_normalize(x):
    return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi


#@jax.jit
def _obs_to_th(obs):
    cos_th, sin_th, thdot = obs
    th = jnp.arctan(sin_th / cos_th)
    return th.reshape((1,)), thdot.reshape((1,))


#@jax.jit
def _th_to_obs(th, thdot):
    cos_th, sin_th = jnp.cos(th), jnp.sin(th)
    next_state = jnp.concatenate([cos_th, sin_th, thdot])
    return next_state


class Pendulum(classic_control.PendulumEnv):

    def __init__(self):
        super(Pendulum, self).__init__()
        high = jnp.array([1.0, 1.0, self.max_speed])
        self.max_angle = 10
        self.max_speed = 8.
        self.action_space = gym.spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=jnp.float32)
        self.observation_space = gym.spaces.Box(low=-high, high=high, dtype=jnp.float32)
        self.viewer = None
        self.state = None
        self.last_u = None
        self.reset()

        def _dynamics(state, action):
            th, thdot = _obs_to_th(state)
            newthdot = (thdot + (-3 * self.g / (2 * self.l) * jnp.sin(th + jnp.pi) + 3.0 / (
                    self.m * self.l ** 2) * action) * self.dt)
            newth = th + newthdot * self.dt
            next_state = _th_to_obs(newth, newthdot)
            return next_state

        def _reward(state, action):
            th, th_dot = _obs_to_th(state)
            cost = jnp.sum(angle_normalize(th) ** 2 + 0.1 * th_dot ** 2 + 0.001 * (action ** 2))
            return -cost

        self.f = jax.jit(_dynamics)
        self.r = jax.jit(_reward)

    def reset(self):
        self.state = jnp.array((1., 0., 0.))
        return self.state

    def step(self, action):
        # _clipped_action  = action
        self.last_u = action  # for rendering
        reward = self.r(self.state, action)
        # action = jnp.clip(action, -self.max_torque, self.max_torque)
        next_state = self.f(self.state, action)
        th, th_dot = _obs_to_th(next_state)
        th_dot = jax.lax.clamp(-self.max_speed,th_dot, self.max_speed)
        #th_dot = jnp.clip(self.state[1], -self.max_speed, self.max_speed)
        th = th + th_dot * self.dt
        next_state = _th_to_obs(th, th_dot)

        self.state = next_state
        return next_state, reward, False, {}  # {"slack_action": slack_action, "slack_model": slack_model}
