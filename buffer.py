import torch
import numpy as np
import warnings


class Buffer:
    def __init__(self, d_state, d_action, size):
        """
        data buffer that holds transitions
        Args:
            d_state: dimensionality of state
            d_action: dimensionality of action
            size: maximum number of transitions to be stored (memory allocated at init)
        """
        # Dimensions
        self.size = size
        self.d_state = d_state
        self.d_action = d_action

        # Main Attributes
        self.states = torch.zeros(size, d_state).float()
        self.actions = torch.zeros(size, d_action).float()
        self.state_deltas = torch.zeros(size, d_state).float()
        self.rewards = torch.zeros(size).float()

        # Other attributes
        self.normalizer = None
        self.ptr = 0
        self.is_full = False

    def setup_normalizer(self, normalizer):
        self.normalizer = normalizer

    def _add(self, buffer, arr):
        n = arr.size(0)
        excess = self.ptr + n - self.size  # by how many elements we exceed the size
        if excess <= 0:  # all elements fit
            a, b = n, 0
        else:
            a, b = n - excess, excess  # we need to split into a + b = n; a at the end and the rest in the beginning
        buffer[self.ptr:self.ptr + a] = arr[:a]
        buffer[:b] = arr[a:]

    def add(self, states, actions, next_states, rewards):
        """
        add transition(s) to the buffer
        Args:
            states: pytorch Tensors of (n_transitions, d_state) shape
            actions: pytorch Tensors of (n_transitions, d_action) shape
            next_states: pytorch Tensors of (n_transitions, d_state) shape
        """
        states, actions, next_states, rewards = [x.clone().cpu() for x in [states, actions, next_states, rewards]]

        state_deltas = next_states - states
        n_transitions = states.size(0)

        assert n_transitions <= self.size

        self._add(self.states, states)
        self._add(self.actions, actions)
        self._add(self.state_deltas, state_deltas)
        self._add(self.rewards, rewards)

        if self.ptr + n_transitions > self.size or self.is_full:
            warnings.warn("Buffer overflow. Rewriting old samples")

        if self.ptr + n_transitions >= self.size:
            self.is_full = True

        self.ptr = (self.ptr + n_transitions) % self.size

        if self.normalizer is not None:
            for s, a, ns, r in zip(states, actions, state_deltas, rewards):
                self.normalizer.add(s, a, ns, r)

    def add_trajectory(self, traj):
        states = torch.stack([transition[0] for transition in traj])
        actions = torch.stack([transition[1] for transition in traj])
        rewards = torch.stack([transition[2] for transition in traj])
        next_states = torch.stack([transition[3] for transition in traj])

        self.add(states, actions, next_states, rewards)

    def view(self):
        n = len(self)

        s = self.states[:n]
        a = self.actions[:n]
        s_delta = self.state_deltas[:n]
        ns = s + s_delta

        return s, a, ns, s_delta

    def train_batches(self, batch_size):
        """
        return an iterator of batches
        Args:
            batch_size: number of samples to be returned
        Returns:
            state of size (n_samples, d_state)
            action of size (n_samples, d_action)
            next state of size (n_samples, d_state)
        """
        num = len(self)
        indices = np.random.permutation(range(num))

        for i in range(0, num, batch_size):
            j = min(num, i + batch_size)

            if (j - i) < batch_size and i != 0:
                # drop the last incomplete batch
                return

            batch_size = j - i
            batch_indices = indices[i:j]

            states = self.states[batch_indices]
            actions = self.actions[batch_indices]
            state_deltas = self.state_deltas[batch_indices]
            next_states = states + state_deltas

            yield states, actions, next_states

    def train_batches_reward(self, batch_size):
        """
        return an iterator of batches
        Args:
            batch_size: number of samples to be returned
        Returns:
            state of size (n_samples, d_state)
            action of size (n_samples, d_action)
            reward of size (n_samples,)
        """
        num = len(self)
        indices = np.random.permutation(range(num))

        for i in range(0, num, batch_size):
            j = min(num, i + batch_size)

            if (j - i) < batch_size and i != 0:
                # drop the last incomplete batch
                return

            batch_size = j - i
            batch_indices = indices[i:j]

            states = self.states[batch_indices]
            actions = self.actions[batch_indices]
            rewards = self.rewards[batch_indices]

            yield states, actions, rewards

    def __len__(self):
        return self.size if self.is_full else self.ptr

    def __getstate__(self):
        return self.__dict__.copy()