import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from envs.pendulum import State

import torch
import torch.distributions as torch_dist


class CartPole:
    def __init__(self, horizon):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        v = 0 # 1e-3
        p = 1e-3
        a = 1e-3
        self.viewer = None
        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max, ],
            dtype=np.float32,
        )

        def reward(state, action):
            alive_bonus = 10
            x, x_dot, theta, theta_dot = state
            pos_penalty = p * (x - 10).sum() ** 2
            vel_penalty = v * x_dot.sum() ** 2
            control_penalty = a * action.sum() ** 2
            diff_bonus = (theta - 0.2) ** 2 * alive_bonus
            r = diff_bonus - vel_penalty - control_penalty - pos_penalty
            return r

        def f(state, action):
            x, x_dot, theta, theta_dot = torch.split(state,1)
            force = action
            costheta = torch.cos(theta)
            sintheta = torch.sin(theta)

            temp = (
                           force + self.polemass_length * theta_dot ** 2 * sintheta
                   ) / self.total_mass
            thetaacc = (self.gravity * sintheta - costheta * temp) / (
                    self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
            )
            xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc

            return torch.cat([x, x_dot, theta, theta_dot])

        self.dynamics = f
        self.reward = reward
        self.horizon = horizon
        self.t= 0

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def __del__(self):
        self.close()

    def render(self, state, mode="human"):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = (
                -polewidth / 2,
                polewidth / 2,
                polelen - polewidth / 2,
                -polewidth / 2,
            )
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(0.8, 0.6, 0.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    @property
    def observation_size(self):
        return 4

    @property
    def action_size(self):
        return 1

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, state, action):
        obs= self.dynamics(state.state, action)
        reward = self.reward(state.state, action)

        x, theta = obs[0], obs[2]

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        done = done or self.t == self.horizon
        done = torch.tensor(done,dtype=torch.float32)
        self.t +=1

        return State(obs, obs, reward, done)

    def reset(self,seed):
        torch.manual_seed(seed)

        #$ pos = torch.tensor((0., ), dtype=torch.float32)
        #$ ang = torch.tensor((np.pi / 2, ), dtype=torch.float32)
        # obs = torch_dist.Uniform(low=-0.05, high=0.05).sample(sample_shape=(4,))
        obs = torch.tensor([0., 0.1, 0.1, 0. ])
        reward, done = torch.zeros(2)
        state = State(
            state=obs,
            obs=obs,
            reward=reward,
            done=done,
        )
        return state
