import torch

from envs.pendulum import State


class Ulqr():
    # https://arxiv.org/pdf/1803.07055.pdf

    @property
    def observation_size(self):
        return 3

    @property
    def action_size(self):
        return 3

    def close(self):
        return

    def __init__(self, horizon=200):
        # state and action space are unbounded.
        self.A = torch.tensor([[1.01, 0.01, 0.], [0.01, 1.01, 0.01], [0, 0.01, 1.01]])
        self.B = torch.eye(3)
        self.horizon = horizon
        self.viewer  = None

        self.Q = 1e-3 * torch.eye(3)
        self.R = torch.eye(3)

        def _reward(state, action):
            cost = (state.T @ self.Q @ state + action.T @ self.R @ action)
            return -cost

        def _dynamics(state, action):
            #  add noise
            return self.A @ state + self.B @ action

        self.dynamics = _dynamics
        self.reward = _reward

        self.S = torch.tensor([[0.04529334, 0.01308373, 0.00140714], [0.01308373, 0.04670048, 0.01308373],
                               [0.00140714, 0.01308373, 0.04529334]])

        self.K = torch.tensor([[0.04373095, 0.01250864, 0.00126936], [0.01250864, 0.04500031, 0.01250864],
                               [0.00126936, 0.01250864, 0.04373095]])

    def _optimal_control(self, state):
        return -self.K @ state

    def reset(self, random_key):
        # self.initial_state = jnp.array([1.7640524, 0.4001572, 0.978738])
        torch.manual_seed(random_key)
        state = torch.randn(size=(self.observation_size,))
        done = torch.tensor(0.)
        self.t = 0
        return State(state, state, torch.tensor(0.), done)

    def render(self, state, mode='human'):
        print(state)

    def step(self, state, action):
        r = self.reward(state.obs, action)
        s1 = self.dynamics(state.obs, action) + torch.randn(size=state.obs.shape)
        done = torch.tensor(0.)
        if self.t == self.horizon or torch.linalg.norm(s1) > 1e2:
            done = torch.tensor(1.)
        self.t += 1
        return State(s1, s1, r, done)
