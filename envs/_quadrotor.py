# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from os import path

import gym
import jax
import jax.numpy as jnp

dissipative = lambda x, y, wind: [wind * x, wind * y]
constant = lambda x, y, wind: [wind * jnp.cos(jnp.pi / 4.0), wind * jnp.sin(jnp.pi / 4.0)]


class Quadrotor(gym.core.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, wind=0.0, wind_func=constant):
        self._max_cost = - 1000
        self.m, self.l, self.g, self.dt, self.wind, self.wind_func = (
            0.1,
            0.2,
            9.81,
            0.05,
            wind,
            wind_func,
        )
        self.initial_state, self.goal_state, self.goal_action = (
            jnp.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            jnp.array([self.m * self.g / 2.0, self.m * self.g / 2.0]),
        )

        self.viewer = None
        self.action_dim, self.state_dim = 2, 6

        self.action_space = gym.spaces.Box(-1, 1, shape=(2,))
        self.observation_space = gym.spaces.Box(-1, 1, shape=(6,))

        @jax.jit
        def wind_field(x, y):
            return self.wind_func(x, y, self.wind)

        def _dynamics(x, u):
            state = x
            x, y, th, xdot, ydot, thdot = state
            u1, u2 = u
            m, g, l, dt = self.m, self.g, self.l, self.dt
            wind = wind_field(x, y)
            xddot = -(u1 + u2) * jnp.sin(th) / m + wind[0] / m
            yddot = (u1 + u2) * jnp.cos(th) / m - g + wind[1] / m
            thddot = l * (u2 - u1) / (m * l ** 2)
            state_dot = jnp.array([xdot, ydot, thdot, xddot, yddot, thddot])
            new_state = state + state_dot * dt
            return new_state

        def _reward(x, u):
            mass = 0.1
            goal_state = jnp.zeros((6,))
            goal_action = jnp.array([mass * 9.81 / 2.0, mass * 9.81 / 2.0])
            return - (0.01 * jnp.linalg.norm(u - goal_action) ** 2 + 0.001 * jnp.linalg.norm(x - goal_state) ** 2)

        self.f = jax.jit(_dynamics)
        self.r = jax.jit(_reward)

    def reset(self):
        self.state, self.last_u, self.h = self.initial_state, None, 0
        return self.state

    def step(self, u):
        self.last_u = u
        reward = self.r(self.state, u)
        next_state = self.f(self.state, u)
        next_state = jnp.clip(next_state, -10, 10)
        self.state = next_state
        done = False
        if reward < self._max_cost:
            done = True
        return next_state, reward, done, {}

    def render(self, mode="human"):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(1000, 1000)
            self.viewer.set_bounds(-1.2, 1.2, -1.2, 1.2)
            fname = path.join(path.dirname(__file__), "assets", "drone.png")
            self.img = rendering.Image(fname, 0.4, 0.17)
            self.img.set_color(1.0, 1.0, 1.0)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)
            fnamewind = path.join(path.dirname(__file__), "assets", "wind.png")
            self.imgwind = rendering.Image(fnamewind, 2.4, 2.4)
            self.imgwind.set_color(0.5, 0.5, 0.5)
            self.imgtranswind = rendering.Transform()
            self.imgwind.add_attr(self.imgtranswind)

        self.viewer.add_onetime(self.imgwind)
        self.viewer.add_onetime(self.img)
        self.imgtrans.set_translation(self.state[0], self.state[1] + 0.04)
        self.imgtrans.set_rotation(self.state[2])
        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
