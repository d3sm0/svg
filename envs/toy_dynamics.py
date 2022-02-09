import gym
import torch


class DynamicsEnv(gym.Env):
    def __init__(self, dt=0.1, horizon=200):
        self.x = torch.zeros((2, 1))
        self.A = torch.tensor([[1, dt], [0, 1]])
        self.B = torch.tensor([[0], [dt]])
        self.action_space = gym.spaces.Box(-1, 1, shape=(1,), )
        self.observation_space = gym.spaces.Box(-1, 1, shape=(2,), )
        self.dt = dt
        self.t = 0
        self.horizon = horizon

    def step(self, action: torch.Tensor):
        self.x = self.A @ self.x + self.B @ action
        done = torch.tensor(1.) if self.t >= self.horizon else torch.tensor(0.)
        self.t += self.dt
        return self.x, self.reward(), done, {}

    def reward(self):
        return torch.relu(1 - self.x[0, 0] ** 2)

    def reset(self):
        self.x = torch.tensor([[-1.], [0.]])
        return self.x
