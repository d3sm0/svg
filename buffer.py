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

class ReplayMemory:

    def __init__(self, capacity):
        self.memory = collections.deque([],maxlen=capacity)

    def append(self, transition:Transition):
        """Save a transition"""
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Trajectory:
    def __init__(self):
        self._data = []

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __repr__(self):
        return f"N:{self.__len__()}"

    def append(self, transition):
        self._data.append(transition)

    def get_trajectory(self):
        return  self._data

    def sample_partial(self, horizon):
        assert self.__len__() - horizon - 1 > 0, "not enough data"
        idx = torch.randint(self.__len__() - horizon - 1, (1,))
        return self._data[idx:idx + horizon]

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
            noise= torch.stack(noise)
            yield Transition(s, a, r, s1, done,noise)

