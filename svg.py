import torch
from torch._vmap_internals import vmap

import config
import utils


def recreate_transition(state, transition, dynamics, policy):
    # this function makes the gradient along the a trajectory equivalent regardless the parameters of the function
    mu, std = policy(state)
    eps = (transition.action - mu) / std
    action = mu + std * eps.detach()

    s1_hat = dynamics(state, action)
    eta = (transition.next_state - s1_hat)
    next_state = s1_hat + eta.detach()

    return action, next_state


def verify_transition(transition, state, action, reward, next_state):
    assert torch.allclose(transition.state, state)
    assert torch.allclose(transition.next_state, next_state)
    assert torch.allclose(transition.action, action)
    assert torch.allclose(transition.reward, reward)


def unroll(trajectory, agent, model, gamma):
    total_return = torch.tensor(0.)
    state = trajectory[0].state
    for t, transition in enumerate(trajectory):
        # Remember as the policy drift  the action drift and thus the visited state
        mu, std = agent.forward(state)
        action = mu + std * transition.noise
        # action = action + (transition.action - action)
        next_state = model.dynamics(state, action) + (transition.next_state - model.dynamics(state, action))
        reward = model.reward(state, action)
        verify_transition(transition, state, action, reward, next_state)
        # action, next_state = recreate_transition(transition, policy)
        # Assume we get the reward upon transition to s1 and not at s1
        total_return += (gamma ** t) * reward
        state = next_state
    # we do not change the value function here, but we need to backprop throuhg
    total_return += (1 - transition.done) * agent.value(state).squeeze() * gamma ** config.horizon
    return total_return


def one_step(transition, agent, model, gamma):
    mu, std = agent.forward(transition.state)
    action = mu + std * transition.noise
    r = vmap(model.reward)(transition.state, action)
    next_state = vmap(model.dynamics)(transition.state, action)
    # TODO missing IW if policy drifts off policy
    value = r + gamma * (1 - transition.done) * agent.value(next_state).squeeze()
    return value, {
        "std": torch.linalg.norm(std, 1).mean()
    }


def actor(replay_buffer, agent, model, pi_optim, batch_size=32, gamma=0.99):
    total_loss = torch.tensor(0.)
    n_samples = 0
    pi_optim.zero_grad()
    for transition in replay_buffer.sample(batch_size=batch_size):
        value, metrics = one_step(transition, agent, model, gamma=gamma)
        (-value).mean().backward()
        total_loss += value.mean().detach()
        n_samples += 1
    grad_norm = utils.get_grad_norm(agent.actor.parameters())
    pi_optim.step()
    pi_optim.zero_grad()
    return {
        "actor/loss": total_loss.detach() / n_samples,
        "actor/grad_norm": grad_norm.detach(),
        "actor/std": metrics.get("std")
    }


def critic(repay_buffer, agent, pi_optim, batch_size=32, gamma=0.99):
    # TODO n-step return here
    total_loss = torch.tensor(0.)
    n_samples = 0
    agent.zero_grad()
    for (s, a, r, s1, done, noise) in repay_buffer.sample(batch_size):
        loss = td_loss(agent, s, r, s1, done, gamma).mean()
        loss.mean().backward()
        total_loss += loss
        n_samples += 1
    pi_optim.step()
    grad_norm = utils.get_grad_norm(agent.critic.parameters())

    total_loss = total_loss / n_samples
    return {
        "critic/td": total_loss.detach(),
        "critic/grad_norm": grad_norm.detach(),
    }


def td_loss(policy, s, r, s1, done, gamma):
    td = r + gamma * (1 - done) * policy.value(s1).squeeze() - policy.value(s).squeeze()
    loss = (0.5 * (td ** 2))
    return loss
