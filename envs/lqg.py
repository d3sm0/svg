
import gym
import numpy as np
from gym import register


class Lqg(gym.core.Env):
    action_dim = 1
    state_dim = 2

    def __init__(self):
        action_dim = 1
        state_dim = 2
        self.action_space = gym.spaces.Box(-1, 1, shape=(action_dim,))
        self.observation_space = gym.spaces.Box(-1, 1, shape=(state_dim,))

        self.initial_state = np.zeros(state_dim)

        # Dynamics
        self.A = np.array([[1., 0.], [0., 1.]])
        self.B = np.array([[0.3],
                           [-0.3]])
        self.AB = np.hstack((self.A, self.B))

        # Cost
        self.Q = 0.1 * np.eye(state_dim)
        self.R = np.array([[1]])

        def _reward(state, action):
            cost = (state.transpose() @ self.Q @ state + action.transpose() @ self.R @ action)
            return -cost

        def _dynamics(state, action):
            return self.AB @ np.concatenate((state, action))

        self.dynamics = _dynamics
        self.reward = _reward

    def reset(self):
        self.state = self.initial_state
        return self.state

    def render(self, mode='human'):
        print(self.state)

    def step(self, action):
        r = self.reward(self.state, action)
        s1 = self.dynamics(self.state, action)
        self.state = s1
        return s1, r, False, {}

