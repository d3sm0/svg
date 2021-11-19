import dataclasses
import operator

import torch

import collections
import random


@dataclasses.dataclass
class Transition:
    state: torch.tensor
    action: torch.tensor
    reward: torch.tensor
    next_state: torch.tensor
    done: torch.tensor
    noise: torch.tensor

    def __iter__(self):
        for attr in ["state", "action", "reward", "next_state", "done", "noise"]:
            yield getattr(self, attr)


class ReplayBuffer:
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
        return Transition(torch.stack(states), torch.stack(actions), torch.stack(rewards), torch.stack(next_states), torch.stack(dones), torch.stack(noises))

    def sample(self, batch_size):
        idxes =torch.randint(len(self._storage) - 1,(batch_size,))
        return self._encode_sample(idxes)
