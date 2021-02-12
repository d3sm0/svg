import importlib
from typing import NamedTuple

import gym
import jax
import jax.numpy as jnp


# from utils import RunningMeanStd


class Wrapper(gym.Wrapper):
    def __init__(self, env, key, horizon):
        super(Wrapper, self).__init__(env)
        self.env_spec = EnvSpec(self.observation_space.shape[0], self.action_space.shape[0])
        self.key = key
        self.returns = 0
        self.t = 0
        self.horizon = horizon

    def step(self, action):
        action = jax.lax.clamp(-1., action, 1.)
        next_state, r, d, _info = super(Wrapper, self).step(action)
        self.t += 1
        self.returns += r
        self.key, key = jax.random.split(self.key)
        model_noise = jnp.zeros_like(next_state)
        # model_noise = jax.random.normal(key) * config.model_std
        # next_state = next_state #+ model_noise
        # next_state = jax.lax.clamp(-10., next_state, 10.)
        # next_state = (next_state - 10.) / 20
        # self.unwrapped.state = next_state
        info = {"model_noise": model_noise, "returns": self.returns, "steps": self.t, **_info}
        if self.t >= self.horizon or d is True:
            d = True
        return next_state, r, d, info

    def get_state(self):
        self.returns = 0
        if self.t == self.horizon or self.t == 0:
            self.reset()
        return self.unwrapped.state, 0, False, {}

    def reset(self, **kwargs):
        out = super(Wrapper, self).reset()
        # TODO choose what to do with initial condition
        self.returns = 0
        self.t = 0
        return out, 0., False, {}

    def get_initial_state(self):
        return jnp.zeros((self.env_spec.observation_space,)), jnp.zeros((self.env_spec.action_space,))


class Normalize(gym.Wrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(self, env, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
        super(Normalize, self).__init__(env)
        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.action_rms = RunningMeanStd(shape=self.action_space.shape)
        self.ret_rms = RunningMeanStd(shape=())
        self.clipob = clipob
        self.cliprew = cliprew
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action):
        # action = np.array(action)
        # self.action_rms.update(action)
        # action  = self.action_rms.normalize(action)
        # self.ob_rms.update(self.unwrapped.state)
        obs, rews, news, infos = super(Normalize, self).step(action)
        # self.ret = self.ret * self.gamma + rews
        # obs = obs / (jnp.linalg.norm(obs) + 1e-6)
        # obs = self._obfilt(obs)
        # if self.ret_rms:
        #    self.ret_rms.update(self.ret)
        #    rews = jnp.clip(rews / jnp.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        # self.ret[news] = 0.
        return obs, rews, news, infos

    def _obfilt(self, obs):
        new_obs = self.ob_rms.normalize(obs)
        return new_obs

    def reset(self):
        out = super(Normalize, self).reset()
        self.ob_rms.reset_statistics()
        self.ret_rms.reset_statistics()
        return out

        # self.ret = jnp.zeros(self.num_envs)
        # obs = self.venv.reset()
        # return self._obfilt(obs)


class EnvSpec(NamedTuple):
    observation_space: int
    action_space: int


def make_env(key, env_id="", horizon=100):
    fname = f"envs.{env_id}"
    Env = getattr(importlib.import_module(fname), env_id.capitalize())
    env = Env()
    return Wrapper(env, key, horizon=horizon)


def _test_env():
    env_list = ["acrobot"]  # , "lqg", "pendulum", "quadrotor"]
    key = jax.random.PRNGKey(0)

    # env = monitor.Monitor(env, directory=config.log_dir)

    def _eval_env(env_id):
        env = make_env(key=key, env_id=env_id, horizon=10)
        env.reset()
        while True:
            s1, r, d, _ = env.step(env.action_space.sample())
            env.render()
            if d:
                break
        env.close()

    for env_id in env_list:
        _eval_env(env_id)


if __name__ == "__main__":
    _test_env()
