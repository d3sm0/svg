import gym

import torch

T = torch.Tensor


class TorchWrapper(gym.Wrapper):
    def step(self, action: T):
        if self.should_reset:
            return self.reset()
        s, r, d, info = super(TorchWrapper, self).step(action.numpy())
        self.cumulative_reward += r
        if d:
            info = {
                "step": self.t,
                "return": self.cumulative_reward
            }
            self.should_reset = True
        self.t += 1
        return torch.from_numpy(s).to(torch.float32), torch.tensor(r, dtype=torch.float32), torch.tensor(d,
                                                                                                         dtype=torch.float32), info

    def reset(self, **kwargs):
        s = super(TorchWrapper, self).reset()
        self.t = 0
        self.cumulative_reward = 0
        self.should_reset = False
        return torch.from_numpy(s).to(torch.float32)


class GymWrapper(gym.Wrapper):
    def step(self, action: T):
        if self.should_reset:
            return self.reset()
        s, r, d, info = super(GymWrapper, self).step(action)
        self.cumulative_reward += r
        if d:
            info = {
                "step": self.t,
                "return": self.cumulative_reward
            }
            self.should_reset = True
        self.t += 1
        return s, r, d, info

    def reset(self, **kwargs):
        s = super(GymWrapper, self).reset()
        self.t = 0
        self.cumulative_reward = 0
        self.should_reset = False
        return s


def gradient_norm(model: torch.nn.Module) -> torch.Tensor:
    total_norm = torch.tensor(0.)
    for p in model.parameters():
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def params_norm(model: torch.nn.Module) -> torch.Tensor:
    total_norm = torch.tensor(0.)
    for p in model.parameters():
        param_norm = p.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm
