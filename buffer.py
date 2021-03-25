import operator
from typing import NamedTuple

import torch


class Transition(NamedTuple):
    state: torch.tensor
    action: torch.tensor
    reward: torch.tensor
    next_state: torch.tensor
    done: torch.tensor


class Buffer:
    def __init__(self, max_size=int(1e4)):
        """
        data buffer that holds transitions
        Args:
            d_state: dimensionality of state
            d_action: dimensionality of action
            size: maximum number of transitions to be stored (memory allocated at init)
        """
        # Dimensions
        self.max_size = max_size
        self.n_trajectories = 0
        self.n_samples = 0
        self._buffer = []

    def add(self, trajectory):
        """
        add transition(s) to the buffer
        Args:
            states: pytorch Tensors of (n_transitions, d_state) shape
            actions: pytorch Tensors of (n_transitions, d_action) shape
            next_states: pytorch Tensors of (n_transitions, d_state) shape
        """
        if len(self._buffer) > self.max_size:
            t = self._buffer.pop(0)
            self.n_trajectories -= 1
            self.n_samples -= len(t)
        self._buffer.append(trajectory)
        self.n_trajectories += 1
        self.n_samples += len(trajectory)

    def train_batches(self, batch_size, n_epochs=1):
        """
        return an iterator of batches
        Args:
            batch_size: number of samples to be returned
        Returns:
            state of size (n_samples, d_state)
            action of size (n_samples, d_action)
            next state of size (n_samples, d_state)
        """

        for _ in range(n_epochs):
            traj_idx = torch.randint(self.n_trajectories, size=(1,)).item()
            yield self._buffer[traj_idx]

    def __len__(self):
        return len(self._buffer)


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

    def sample_partial(self, horizon):
        assert self.__len__() - horizon - 1 > 0, "not enough data"
        idx = torch.randint(self.__len__() - horizon - 1, (1,))
        return self._data[idx].state, iter(self._data[idx:idx + horizon])

    def sample_batch(self, batch_size, shuffle=False):
        idxs = torch.arange(self.__len__())
        if shuffle:
            idxs = idxs[torch.randperm(self.__len__())]

        idxs = torch.split(idxs, split_size_or_sections=batch_size)
        for idx in idxs:
            batch = operator.itemgetter(*idx)(self._data)
            s, a, r, s1, *_ = list(zip(*batch))
            s = torch.stack(s)
            a = torch.stack(a)
            s1 = torch.stack(s1)
            r = torch.stack(r)
            yield s, a, r, s1
