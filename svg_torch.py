import operator
from typing import NamedTuple

import torch
import torch.nn as nn


class Transition(NamedTuple):
    state: torch.tensor
    action: torch.tensor
    reward: torch.tensor
    next_state: torch.tensor
    done: torch.tensor


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

    def sample(self, batch_size, shuffle=False):
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


def get_grad_norm(parameters):
    return torch.norm(torch.cat([p.grad.flatten() for p in parameters]))


def recreate_transition(state, transition, dynamics, policy):
    a_hat, scale = policy(state)
    eps = (transition.action - a_hat) / scale
    action = a_hat + scale * eps.detach()

    s1_hat, sigma = dynamics(state, action)
    eta = (transition.next_state - s1_hat) / sigma
    next_state = s1_hat + sigma * eta.detach()

    return action, next_state, (scale, sigma)


def unroll(dynamics, policy, traj, gamma=0.99, h_detach=0.0, action_reg=1e-1):
    total_reward = 0
    state = traj[0].state
    noises = []
    for t, transition in enumerate(traj):
        action, next_state, noise = recreate_transition(state, transition, dynamics, policy)
        noises.append(noise)
        # assert
        reward = dynamics.reward(state, action)  # - action_reg * action.norm() ** 2
        total_reward += (gamma ** t) * reward
        # if torch.rand((1,)) < h_detach:
        #    next_state.detach_()
        state = next_state
    a, b = list(zip(*noises))
    sigma_avg = torch.tensor(a).mean()
    eta_avg = torch.tensor(b).mean()
    return total_reward / len(traj), {"agent/pi_scale": sigma_avg, "agent/eta_scale": eta_avg}


def train(dynamics, policy, pi_optim, traj, grad_clip=5., gamma=0.99):
    total_return, extra = unroll(dynamics, policy, traj, gamma=gamma, action_reg=0., h_detach=0.)
    pi_optim.zero_grad()
    (-total_return).backward()
    true_norm = get_grad_norm(parameters=policy.parameters())
    nn.utils.clip_grad_value_(policy.parameters(), grad_clip)
    pi_optim.step()
    return {"agent/return": total_return, "agent/grad_norm": true_norm, **extra}


def generate_episode(env, policy):
    state, _, done, info = env.reset()
    trajectory = Trajectory()
    while not done:
        action = policy.sample(state)
        next_state, reward, done, info = env.step(action)
        trajectory.append(Transition(state, action, reward, next_state, done))
        state = next_state
    return trajectory, info


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
