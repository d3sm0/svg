import operator
from typing import NamedTuple
import config

import torch
import torch.nn as nn

from buffer import Trajectory, Transition


def get_grad_norm(parameters):
    grad_norm = 0
    for p in parameters:
        if p.grad is not None:
            grad_norm += p.norm().cpu()
    return grad_norm


def l2_norm(parameters):
    grad_norm = 0
    for n, p in parameters:
        if "bias" in n:
            continue
        grad_norm += p.norm().cpu()
    return grad_norm


def recreate_transition(state, transition, dynamics, policy):
    a_hat, scale = policy(state)
    assert scale > 0
    eps = (transition.action - a_hat) / scale
    action = a_hat + scale * eps.detach()

    s1_hat, sigma = dynamics(state, action)
    assert (sigma > 0).all()
    eta = (transition.next_state - s1_hat) / sigma
    next_state = s1_hat + sigma * eta.detach()

    return action, next_state, (scale.norm(), sigma.norm())


def unroll(dynamics, policy, traj, state):
    total_return = torch.zeros((1,))
    noises = []
    for t, transition in enumerate(traj):
        action, next_state, noise = recreate_transition(state, transition, dynamics, policy)
        noises.append(noise)
        reward = dynamics.reward(state, action)
        total_return += (config.gamma ** t) * reward
        next_state = next_state
        state = next_state
    # TODO test vf detach
    total_return += policy.value(state).detach() * config.gamma ** config.horizon
    return total_return, {"scale": noise[0], "sigma": noise[1]}


def train(dynamics, policy, pi_optim, trajectory, model_optim):
    total_td = 0
    for _ in range(config.opt_epochs):
        for (s, a, r, s1) in trajectory.sample_batch(batch_size=config.batch_size):
            td = r + policy.value(s1).squeeze() - policy.value(s).squeeze()
            loss = (0.5 * (td ** 2)).mean()
            total_td += loss
            pi_optim.zero_grad()
            loss.backward()
            pi_optim.step()
        total_td /= len(trajectory)
    pi_optim.zero_grad()

    model_loss = 0
    for _ in range(config.opt_epochs):
        for (s, a, r, s1) in trajectory.sample_batch(batch_size=config.batch_size):
            loss = (dynamics(s, a)[0] - s1).norm(2, 1).pow(2).mean()
            model_loss += loss
            model_optim.zero_grad()
            loss.backward()
            model_optim.step()
        model_loss /= len(trajectory)
    model_optim.zero_grad()

    start_state, partial_trajectory = trajectory.sample_partial(config.train_horizon)
    total_return, extra = unroll(dynamics, policy, partial_trajectory, start_state)
    assert torch.isfinite(total_return)
    pi_optim.zero_grad()
    (-total_return).backward()
    pi_grad_norm = get_grad_norm(parameters=policy.parameters())
    pi_norm = l2_norm(policy.named_parameters())
    torch.nn.utils.clip_grad_value_(policy.parameters(), clip_value=config.grad_clip)
    pi_optim.step()

    return {"agent/return": total_return,
            "agent/td_error": total_td,
            "agent/train": model_loss,
            "agent/grad_norm": pi_grad_norm,
            "agent/pi_norm": pi_norm,
            **extra}


def generate_episode(env, policy):
    state, _, done, info = env.reset()
    trajectory = Trajectory()
    while not done:
        action = policy.sample(state)
        next_state, reward, done, info = env.step(action)
        trajectory.append(Transition(state, action, reward, next_state, done))
        state = next_state
    return trajectory, info
