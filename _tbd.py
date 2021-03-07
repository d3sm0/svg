from torch import nn as nn


def train_model_on_buffer(dynamics, buffer, model_optim, batch_size=64, n_batches=16):
    buffer_it = buffer.train_batches(batch_size)
    total_loss = 0
    for _ in range(n_batches):
        try:
            state, action, next_state = next(buffer_it)
        except StopIteration:
            break
        model_optim.zero_grad()
        mu, _ = dynamics(state, action)
        loss = nn.functional.mse_loss(mu, next_state)
        total_loss += loss
        loss.backward()
        model_optim.step()
    return total_loss / n_batches


def train_reward_on_buffer(dynamics, buffer, reward_optim, batch_size=64, n_batches=16):
    buffer_it = buffer.train_batches_reward(batch_size)
    total_loss = 0
    for _ in range(n_batches):
        try:
            state, action, reward = next(buffer_it)
        except StopIteration:
            break
        reward_optim.zero_grad()
        loss = nn.functional.mse_loss(dynamics.reward(state, action), reward)
        total_loss += loss
        loss.backward()
        reward_optim.step()
    return total_loss / n_batches