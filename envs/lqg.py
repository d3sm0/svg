import gym
import numpy as np
import torch


class Lqg(gym.core.Env):
    action_dim = 1
    state_dim = 2

    def _get_obs(self):
        return self.state

    def __init__(self):
        action_dim = 1
        state_dim = 2
        self.action_space = gym.spaces.Box(-1, 1, shape=(action_dim,))
        self.observation_space = gym.spaces.Box(-1, 1, shape=(state_dim,))

        # self.initial_state = jnp.zeros(state_dim)
        np.random.seed(0)
        self.initial_state = torch.tensor(np.random.normal(size=(2,)), dtype=torch.float32)
        self.A = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float32)
        self.B = torch.tensor([[0.3], [-0.3]], dtype=torch.float32)

        # Cost
        self.Q = 0.1 * torch.eye(state_dim)
        self.R = torch.tensor([[1.]], dtype=torch.float32)

        self.K = torch.tensor([[3.34826643, -3.34826643]])
        self.S = torch.tensor([[5.58044406, -5.58044406], [-5.58044406, 5.58044406]])
        self.state = None

    def reward_fn(self, state, action):
        cost = (state.T @ self.Q @ state + action.T @ self.R @ action)
        return -cost

    def dynamics_fn(self, state, action):
        next_state = self.A @ state + self.B @ action
        return next_state

    def _optimal_control(self, state):
        return -self.K @ state

    def reset(self):
        self.t = 0
        self.state = self.initial_state
        return self.state

    def render(self, mode='human'):
        print(self.state)

    def step(self, action):
        assert torch.linalg.norm(action) < 1e3
        reward = self.reward_fn(self.state, action)
        self.state = self.dynamics_fn(self.state, action)
        done = torch.tensor(1.) if torch.linalg.norm(action) > 1e1 else torch.tensor(0.)
        done = done if self.t < 199 else torch.tensor(1.)
        self.t += 1
        self.state = torch.clamp(self.state, -10., 10)
        return self.state, reward, done, {}
