from typing import NamedTuple

import gym
import torch


class EnvSpec(NamedTuple):
    observation_space: int
    action_space: int


class Wrapper(gym.Wrapper):
    def __init__(self, env, horizon):
        super(Wrapper, self).__init__(env)
        self.env_spec = EnvSpec(self.observation_space.shape[0], self.action_space.shape[0])
        self.returns = 0
        self.t = 0
        self.horizon = horizon
        self.action_bound = 1.
        self._r = env._r
        #self._f = env._f

    def step(self, action):
        action = torch.clamp(action, -self.action_bound, self.action_bound)
        action = action.numpy()
        next_state, r, d, _info = super(Wrapper, self).step(action)
        self.returns += r
        info = {"env/reward": r,
                "env/avg_reward": self.returns / (self.t + 1),
                "env/returns": self.returns,
                "env/steps": self.t}
        if self.t >= self.horizon or d is True:
            d = True
        self.t += 1
        return torch.tensor(next_state), torch.tensor(r), d, info

    def get_state(self):
        self.returns = 0
        if self.t == self.horizon or self.t == 0:
            self.reset()
        return self.unwrapped.state, 0, False, {}

    def reset(self, **kwargs):
        out = super(Wrapper, self).reset()
        self.returns = 0
        self.t = 0
        out = torch.tensor(out, dtype=torch.float32)
        return out

    def get_initial_state(self):
        return torch.zeros((self.env_spec.observation_space,)), torch.zeros((self.env_spec.action_space,))


def make_env(env_id="lqg", horizon=200):
    # env = pendulum.Pendulum()
    import pybullet_envs.gym_pendulum_envs
    pybullet_envs.gym_pendulum_envs.InvertedDoublePendulumBulletEnv()

    # import pybullet_envs
    # pybullet_envs.registry
    # env_id = 'InvertedPendulumSwingupBulletEnv-v0'
    # id = 'CartPoleContinuousBulletEnv-v0'
    # env = gym.make(env_id)
    import envs.pendulum
    env = envs.pendulum.Pendulum()
    env = Wrapper(env, horizon=horizon)
    return env


def _test_env():
    env_list = ["acrobot"]  # , "lqg", "pendulum", "quadrotor"]

    # env = monitor.Monitor(env, directory=config.log_dir)

    def _eval_env(env_id):
        env = make_env(env_id)
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
