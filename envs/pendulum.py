import gym
from gym.envs.classic_control.pendulum import PendulumEnv
import jax
import math
import jax.numpy as jnp
import numpy as np

_env = PendulumEnv()
G, L, M, DT = _env.g, _env.l, _env.m, _env.dt
_MAX_TORQUE = float(_env.max_torque)
_MAX_SPEED = float(_env.max_speed)
del _env


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class DifferentiablePendulum(gym.envs.classic_control.PendulumEnv):
    def __init__(self):
        super(DifferentiablePendulum, self).__init__()

        def _reward(state, action):
            theta_cos, theta_sin, thetadot = jnp.split(state, 3)
            theta = jnp.arctan2(theta_sin, theta_cos)
            clamped_actions = jax.lax.clamp(-_MAX_TORQUE, action, _MAX_TORQUE)

            costs = angle_normalize(theta) ** 2 + 0.1 * thetadot ** 2 + 0.001 * (clamped_actions ** 2)
            rewards = -costs
            return rewards

        def _dynamics_in_stspace(th, thdot, action):
            u = jax.lax.clamp(-_MAX_TORQUE, action, _MAX_TORQUE)

            newthdot = thdot + (-3 * G / (2 * L) * jnp.sin(th + math.pi) + 3. / (M * L ** 2) * u) * DT
            newth = th + newthdot * DT
            newthdot = jax.lax.clamp(-_MAX_SPEED, newthdot, _MAX_SPEED)
            return newth, newthdot

        self._dynamics_in_stspace = _dynamics_in_stspace

        def _dynamics(state, action):
            theta_cos, theta_sin, thdot = jnp.split(state, 3)
            th = jnp.arctan2(theta_sin, theta_cos)

            newth, newthdot = _dynamics_in_stspace(th, thdot, action)
            next_state = jnp.concatenate([jnp.cos(newth), jnp.sin(newth), newthdot])
            return next_state

        self.dynamics = jax.jit(_dynamics)
        self.reward = jax.jit(_reward)

    def reset(self):
        self.last_u = None
        self.state = np.array([0., 0.], dtype=np.float32)
        return self._get_obs()
