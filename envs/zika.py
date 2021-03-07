try:
    from whynot.simulators.zika.environments import ZikaEnv as _ZikaEnv
except ImportError:
    print("please install whynot")
from dataclasses import fields

import torch


# TODO check for rw function
def get_reward(time, state, action, gamma=0.5):
    """Compute the reward based on the observed state and choosen intervention."""
    state_weights = torch.tensor([60., 500., 60.])
    # asymptomatic humans, symptomatic humans, mosquitos
    state_indeces = torch.tensor([1, 2, -1])

    action_weights = torch.tensor([25., 20., 30., 40.])
    # treated_bednet_use, condom_use, treatment_of_infected, indoor_spray_use
    action_indeces = torch.tensor([0, 1, 2, 3])

    state_cost = (state_weights * state[state_indeces]).sum()
    action_cost = (action_weights * action[action_indeces] ** 2).sum()
    return -(state_cost + 0.5 * action_cost) * torch.exp(-gamma * time)


class ZikaEnv(_ZikaEnv):
    def __init__(self, gamma=0.5):
        super(ZikaEnv, self).__init__()
        self.reward = get_reward
        self.gamma = gamma

    def step(self, action):
        # Compute reward on clipped action
        current_state = torch.tensor([field.default for field in fields(self.env.state)])
        r = self.reward(current_state, action)
        next_state, _, d, info = self.env.step(action)
        return next_state, r, d, info
