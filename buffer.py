import operator
from typing import NamedTuple

import torch

import collections
import random


class Transition(NamedTuple):
    state: torch.tensor
    action: torch.tensor
    reward: torch.tensor
    next_state: torch.tensor
    done: torch.tensor
    noise: torch.tensor


class ReplayBuffer(object):
    def __init__(self, size):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def append(self, transition: Transition):

        if self._next_idx >= len(self._storage):
            self._storage.append(transition)
        else:
            self._storage[self._next_idx] = transition
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        states, actions, rewards, next_states, dones, noises = [], [], [], [], [], []

        for i in idxes:
            data = self._storage[i]
            state, action, reward, next_state, done, noise = data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            noises.append(noise)
        return Transition(torch.stack(states), torch.stack(actions), torch.stack(rewards), torch.stack(next_states),
                          torch.stack(dones), torch.stack(noises))

    def sample(self, batch_size):
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class Trajectory:
    def __init__(self):
        self._data = []
        self._values = []

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __repr__(self):
        return f"N:{self.__len__()}"

    def append(self, transition, value):
        self._data.append(transition)
        self._values.append(value)

    def get_trajectory(self):
        return self._data

    def get_returns(self, gamma=0.99):
        discounts = [1.]
        rewards = [0.]
        masks = [0.]
        values = []
        for t, transition in enumerate(self._data):
            discounts.append(gamma ** t)
            rewards.append(transition.reward)
            values.append(self._values[t])
            masks.append(transition.done)

        discounts = [d  * (1-m) for d,m in zip(discounts[1:] , masks[1:])]
        values = values[1:] + [torch.tensor(0.)]
        rewards = rewards[1:]

        from utils import n_step_bootstrapped_returns, lambda_returns
        td_lambda = lambda_returns(torch.stack(rewards), torch.tensor(discounts), torch.stack(values), stop_target_gradients=False)
        # td_lambda = n_step_bootstrapped_returns(torch.stack(rewards), torch.tensor(discounts), torch.stack(values), n=4, stop_target_gradients=False)
        self._values = td_lambda

    def sample_partial(self, horizon):
        # effective_horizon  = min(self.__len__(),horizon  -2 )
        start_idx = torch.randint(self.__len__() - 1, (1,))
        horizon = min(self.__len__() - start_idx, horizon)
        return self._data[start_idx:start_idx + horizon]

    def sample(self, batch_size, shuffle=False):
        idxs = torch.arange(self.__len__())
        if shuffle:
            idxs = idxs[torch.randperm(self.__len__())]

        idxs = torch.split(idxs, split_size_or_sections=batch_size)
        new_idxs = []
        for idx in idxs:
            if len(idx) == 1:
                continue
            new_idxs.append(idx)
        idxs = new_idxs
        for idx in idxs:
            batch = operator.itemgetter(*idx)(self._data)
            s, a, r, s1, done, noise = list(zip(*batch))
            s = torch.stack(s)
            a = torch.stack(a)
            s1 = torch.stack(s1)
            r = torch.stack(r)
            done = torch.stack(done)
            noise = torch.stack(noise)
            td_lambda = operator.itemgetter(*idx)(self._values)
            td_lambda = torch.stack(td_lambda)
            yield Transition(s, a, r, s1, done, noise), td_lambda
