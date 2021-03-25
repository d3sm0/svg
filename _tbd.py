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


def train_model_on_traj(buffer, dynamics, model_optim, batch_size=64, shuffle=True, n_epochs=5):
    total_loss = 0.
    total_reward_loss = 0.
    total_model_loss = 0.
    total_grad_norm = 0.
    for traj in buffer.train_batches(batch_size=batch_size, n_epochs=n_epochs):
        for (state, action, r, next_state) in traj.sample(batch_size=batch_size, shuffle=shuffle):
            model_optim.zero_grad()
            mu, _ = dynamics(state, action)
            model_loss = 0.5 * (mu - next_state).norm(dim=-1) ** 2
            reward_loss = 0.5 * (r - dynamics.reward(state, action)) ** 2
            loss = (model_loss + reward_loss).mean()
            total_model_loss += model_loss.mean()
            total_reward_loss += reward_loss.mean()
            loss.backward()
            grad_norm = get_grad_norm(dynamics.parameters())
            total_grad_norm += grad_norm
            model_optim.step()
            total_loss += loss
    denom = (len(traj) * batch_size)
    return {"model/total_loss": total_loss / denom,
            "model/model_loss": total_model_loss / denom,
            "model/reward_loss": total_reward_loss / denom,
            "model/grad_norm": total_grad_norm / denom
            }
