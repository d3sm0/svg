from dataclasses import replace

import torch
import torch.distributions as torch_dist
from torch._vmap_internals import vmap

import utils
from buffer import Transition


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
    total_return = torch.zeros((1, ))
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
        total_return = total_return + (gamma**t) * transition.reward
    total_return += gamma**T * (1 - transition.done) * agent.value(transition.next_state).squeeze()
    # we do not change the value function here, but we need to backprop throuhg
    # what happen if we replace the value function bootstrapped here with the one from pi^star?
    # this looks like the target of n-step td but off by one factor.
    # assume determintic action for the bootrap
    return total_return, metrics


def td_loss(r, v_tm1, v_t, done, gamma):
    td = r + gamma * (1 - done) * v_t - v_tm1
    loss = (0.5 * (td**2))
    return loss


def q_loss(r, q_tm1, q_t, done, gamma):
    td = r + gamma * (1 - done) * q_t.detach() - q_tm1
    loss = (0.5 * (td**2))
    return loss


def optimize_actor(replay_buffer, agent, model, pi_optim, batch_size=32, gamma=0.99, epochs=1):
    total_loss = torch.tensor(0.)

    for _ in range(epochs):
        transition = replay_buffer.sample(batch_size=batch_size)
        pi_optim.zero_grad()
        value, metrics = agent.get_value_gradient(transition, model, gamma)
        (-value.mean()).backward()
        pi_optim.step()
        total_loss += value.mean().detach()
    total_loss = total_loss / epochs
    grad_norm = utils.get_grad_norm(agent.actor.parameters())
    return {
        "actor/value": total_loss,
        "actor/grad_norm": grad_norm,
        # "actor/std": metrics.get("std")
    }


def optimize_critic(repay_buffer, agent, critic_optim, batch_size=32, gamma=0.99, epochs=1):
    total_loss = torch.tensor(0.)
    for _ in range(epochs):
        transition = repay_buffer.sample(batch_size)
        agent.critic.zero_grad()
        loss = agent.get_value_loss(transition, gamma)
        loss.mean().backward()
        total_loss += loss.mean()
        critic_optim.step()
    grad_norm = utils.get_grad_norm(agent.critic.parameters())
    total_loss = total_loss / epochs
    return {
        "critic/td": total_loss.detach(),
        "critic/grad_norm": grad_norm.detach(),
    }
