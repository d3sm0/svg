import whynot.simulators.zika.environments
from .torch_envs import Wrapper
from dataclasses import fields
import torch


def get_reward(state, action):
    """Compute the reward based on the observed state and choosen intervention."""
    state_weights = torch.tensor([60., 500., 60.])
    state_indeces = torch.tensor([1, 2, -1])  # asymptomatic humans, symptomatic humans, mosquitos

    action_weights = torch.tensor([25., 20., 30., 40.])
    action_indeces = torch.tensor([0, 1, 2, 3])  # treated_bednet_use, condom_use, treatment_of_infected, indoor_spray_use

    state_cost = (state_weights * state[state_indeces]).sum()
    action_cost = (action_weights * action[action_indeces]**2).sum()
    return -(state_cost + 0.5*action_cost)


class ZikaEnv(Wrapper):
    def __init__(self, horizon=100):
        env = whynot.simulators.zika.environments.ZikaEnv
        env._r = get_reward
        super().__init__(env, horizon)

    def step(self, action):
        action = torch.clamp(action, 0, 1)
        # Compute reward on clipped action
        current_state = torch.tensor([field.default for field in fields(self.env.state)])
        r = self.env._r(current_state, action)
        action = action.numpy()
        next_state, _, d, _info = self.env.step(action)

        self.returns += r
        info = {"env/reward": r,
                "env/avg_reward": self.returns / (self.t + 1),
                "env/returns": self.returns,
                "env/steps": self.t}
        if self.t >= self.horizon or d is True:
            d = True
        self.t += 1
        return torch.tensor(next_state, dtype=torch.float32), r, d, info

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
