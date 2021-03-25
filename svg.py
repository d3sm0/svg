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
        next_state = (next_state * 0.5 + (1 - next_state).detach())
        state = next_state
    # TODO test vf detach
    total_return += policy.value(state) * config.gamma ** config.horizon
    return total_return, {"scale": noise[0], "sigma": noise[1]}


def train(dynamics, policy, pi_optim, trajectory):
    extra = {}
    for _ in range(config.opt_epochs):
        start_state, partial_trajectory = trajectory.sample_partial(config.train_horizon)
        total_return, extra = unroll(dynamics, policy, partial_trajectory, start_state)
        assert torch.isfinite(total_return)
        pi_optim.zero_grad()
        (-total_return).backward()
        pi_grad_norm = get_grad_norm(parameters=policy.parameters())
        model_grad_norm = get_grad_norm(parameters=dynamics.parameters())
        pi_norm = l2_norm(policy.named_parameters())
        model_norm = l2_norm(dynamics.named_parameters())
        (config.regularizer * (pi_norm + model_norm)).backward()
        assert torch.isfinite(pi_grad_norm) and torch.isfinite(model_grad_norm)
        torch.nn.utils.clip_grad_value_(policy.parameters(), clip_value=config.grad_clip)
        torch.nn.utils.clip_grad_value_(dynamics.parameters(), clip_value=config.grad_clip)
        pi_optim.step()

    total_td = 0
    for (s, a, r, s1) in trajectory.sample_batch(batch_size=config.batch_size):
        td = r + policy.value(s1) - policy.value(s)
        loss = (0.5 * (td ** 2)).mean()
        total_td += loss
        pi_optim.zero_grad()
        loss.backward()
        pi_optim.step()
    total_td /= len(trajectory)
    return {"agent/return": total_return, "agent/td_error": total_td, "agent/grad_norm": pi_grad_norm,
            "agent/dynamic_norm": model_grad_norm,
            "agent/model_norm": model_norm,
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
