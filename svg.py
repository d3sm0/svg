from dataclasses import replace

import torch
import torch.distributions as torch_dist
from torch._vmap_internals import vmap

import utils


def verify_transition(transition, state, action, reward, next_state):
    assert torch.allclose(transition.state, state)
    assert torch.allclose(transition.next_state, next_state)
    assert torch.allclose(transition.action, action)
    assert torch.allclose(transition.reward, reward)


def one_step(transition, agent, model):
    mu, std = agent.forward(transition.state)
    action = mu + std * transition.noise.reshape_as(mu).detach()
    h = torch_dist.Normal(mu, std).entropy().mean()
    reward = vmap(model.reward)(transition.state, action)
    next_state = vmap(model.dynamics)(transition.state, action)
    return Transition(next_state, action, reward, next_state, done=transition.done, noise=transition.noise), {
        "std": std,
        "entropy": h
    }


def unroll_trajectory(trajectory, agent, model, gamma):
    total_return = torch.zeros((1,))
    transition = trajectory[0]
    T = len(trajectory)
    metrics = {}
    for t, real_transition in enumerate(trajectory):
        real_transition = trajectory[t]
        transition = replace(real_transition,
                             noise=real_transition.noise,
                             done=real_transition.done,
                             next_state=transition.next_state)
        transition = Transition(*list(map(lambda x: x.unsqueeze(0), transition.__dict__.values())))
        transition, metrics = one_step(transition, agent, model)
        # Remember as the policy drift  the action drift and thus the visited state
        # Assume we get the reward upon transition to s1 and not at s1
        total_return = total_return + (gamma ** t) * transition.reward
    total_return += gamma ** T * (1 - transition.done) * agent.value(transition.next_state).squeeze()
    # we do not change the value function here, but we need to backprop throuhg
    # what happen if we replace the value function bootstrapped here with the one from pi^star?
    # this looks like the target of n-step td but off by one factor.
    # assume determintic action for the bootrap
    return total_return, metrics