import torch


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
