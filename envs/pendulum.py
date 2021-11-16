import typing

import numpy as np
import torch
import torch.distributions as torch_dist


from os import path
def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def _obs_to_th(obs):
    cos_th, sin_th, thdot = obs
    th = torch.atan2(sin_th, cos_th)
    return th.reshape((1,)), thdot.reshape((1,))


def _th_to_obs(th, thdot):
    cos_th, sin_th = torch.cos(th), torch.sin(th)
    next_state = torch.cat([cos_th, sin_th, thdot])
    return next_state


class State(typing.NamedTuple):
    state: torch.tensor
    obs: torch.tensor
    reward: torch.tensor
    done: torch.tensor


class Pendulum():
    state_dim = 3
    action_dim = 1

    def __init__(self,horizon=100):

        self.max_speed = 8
        self.max_torque = 2.0
        self.dt = 0.05
        self.g = 10.
        self.m = 1.0
        self.l = 1.0
        self.max_speed = 8.
        self.viewer = None
        self.state = None
        self.last_u = None
        self.horizon =horizon

        def _dynamics(obs, action):
            th, thdot = _obs_to_th(obs)
            newthdot = (thdot + (-3 * self.g / (2 * self.l) * torch.sin(th + np.pi) + 3.0 / (
                    self.m * self.l ** 2) * action) * self.dt)
            newth = th + newthdot * self.dt
            newthdot = torch.clamp( newthdot, -self.max_speed, self.max_speed)
            next_state = _th_to_obs(newth, newthdot)
            return next_state

        def _reward(obs, action):
            th, th_dot = _obs_to_th(obs)
            cost = torch.sum(angle_normalize(th) ** 2 + 0.1 * th_dot ** 2 + 0.001 * (action ** 2))
            return -cost

        self.dynamics = _dynamics
        self.reward = _reward
        self.t = 0
        self.last_u=None


    def render(self, mode="human"):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, 0.2)
            rod.set_color(0.8, 0.3, 0.3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(0.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1.0, 1.0)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    @property
    def observation_size(self) -> int:
        return self.state_dim

    @property
    def action_size(self) -> int:
        return self.action_dim

    def reset(self, seed):
        self.t = 0
        torch.random.manual_seed(seed)
        pos = torch_dist.Uniform(0.,1.).sample((1,))
        ang = torch_dist.Uniform(low=-np.pi, high=np.pi).sample((1,))
        obs = _th_to_obs(pos, ang)
        _state = torch.cat([pos, ang])
        reward, done, zero = torch.zeros(3)
        self.state = _state
        return State(_state, obs, reward, done)

    def step(self, state: State, action: torch.tensor) -> State:
        # action =  torch.clamp(action,-2., 2.)
        r = self.reward(state.obs, action)
        next_obs = self.dynamics(state.obs, action)
        pos, ang = _obs_to_th(next_obs)
        _state = torch.cat([pos, ang])
        done =0.
        if self.t ==self.horizon:
            done = 1.
        done = torch.tensor(done, dtype=torch.float32)
        self.t +=1
        return State(state=_state, obs=next_obs, reward=r, done=done)
