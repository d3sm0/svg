from dataclasses import replace

import torch
import torch.distributions as torch_dist
from torch._vmap_internals import vmap

import config
import utils
from buffer import Transition, Trajectory


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


def one_step(transition, agent, model):
    mu, std = agent.forward(transition.state)
    action = mu + std * transition.noise.detach()
    h = torch_dist.Normal(mu, std).entropy().mean()
    next_state = vmap(model.dynamics)(transition.state, action)
    reward = vmap(model.reward)(transition.next_state, action)
    # TODO maybe missing IW if policy drifts off policy
    # verify_transition(transition, state, action, reward, next_state)
    return Transition(next_state, action, reward, next_state, done=transition.done, noise=transition.noise), {
        "std": std, "entropy": h}


def _unroll(trajectory, agent, model, gamma):
    total_return = torch.zeros((1,))
    transition = trajectory[0]
    T = len(trajectory)
    metrics = {}
    transition = Transition(*list(map(lambda x: x.requires_grad_(True), transition)))
    for t, real_transition in enumerate(trajectory):
        real_transition = trajectory[t]
        transition = replace(real_transition, noise=real_transition.noise, done=real_transition.done,
                             next_state=transition.next_state)
        transition = Transition(*list(map(lambda x: x.unsqueeze(0), transition.__dict__.values())))
        transition, metrics = one_step(transition, agent, model)
        # Remember as the policy drift  the action drift and thus the visited state
        # Assume we get the reward upon transition to s1 and not at s1
        total_return = total_return + (gamma ** t) * transition.reward

    # extrapolate(agent, model, transition, gamma ** (T + 1) * (1-transition.done))
    total_return += (1 - transition.done) * agent.value(transition.next_state).squeeze() * gamma ** T
    # we do not change the value function here, but we need to backprop throuhg
    # what happen if we replace the value function bootstrapped here with the one from pi^star?
    # this looks like the target of n-step td but off by one factor.
    # assume determintic action for the bootrap
    return total_return, transition, metrics


import torch.optim as optim


def extrapolate(agent, model, transition, gamma):
    local_opt = optim.SGD(agent.critic.parameters(), lr=1e-2)
    transition = Transition(*list(map(lambda x: x.repeat(config.batch_size, 1), transition)))
    for _ in range(config.extrapolation_epochs):
        local_opt.zero_grad()
        with torch.no_grad():
            mu, std = agent(transition.next_state.detach())
            next_action = mu + std * torch.randn(size=(config.batch_size, 1))
            r_t = vmap(model.reward)(transition.next_state, next_action)
            f_t = vmap(model.dynamics)(transition.next_state, next_action)
        v_t = agent.value(f_t).squeeze()
        v_tm = agent.value(transition.next_state.detach()).squeeze()
        td = (r_t + gamma * v_t - v_tm)
        error = 0.5 * td ** 2
        error.mean().backward()
        local_opt.step()


def _unroll_one_step(transition: Transition, agent, model, gamma):
    # this is only for simplicity
    transition, metrics = one_step(transition, agent, model)
    one_step_return = transition.reward + gamma * agent.value(transition.next_state)
    return one_step_return, transition, metrics


def unroll(trajectory, agent, model, gamma):
    if len(trajectory) == 1:
        # here we can use larger batch size. sampler does not support T x N
        return _unroll_one_step(trajectory[0], agent, model, gamma)
    else:
        return _unroll(trajectory, agent, model, gamma)


# actor trajectory and actory batch should just be one function
def actor_trajectory(replay_buffer: Trajectory, agent, model, pi_optim, horizon, gamma=0.99, epochs=1):
    total_loss = torch.tensor(0.)
    for _ in range(epochs):
        pi_optim.zero_grad()
        trajectory = replay_buffer.get_trajectory()
        value, _, _ = unroll(trajectory, agent, model, gamma)
        (-value.mean()).backward()
        total_loss += value.mean().detach()
        pi_optim.step()
    total_loss = total_loss.detach() / epochs
    grad_norm = utils.get_grad_norm(agent.actor.parameters())
    return {
        "actor/value": total_loss,
        "actor/grad_norm": grad_norm,
        # "actor/std": metrics.get("std")
    }


def actor(replay_buffer, agent, model, pi_optim, batch_size=32, gamma=0.99, epochs=1):
    total_loss = torch.tensor(0.)
    n_samples = 0
    for _ in range(epochs):
        for transition in replay_buffer.sample(batch_size=batch_size):
            pi_optim.zero_grad()
            value, _, _ = unroll([transition], agent, model, gamma)
            (-value.mean()).backward()
            pi_optim.step()
            total_loss += value.mean().detach()
            n_samples += 1
    total_loss = total_loss / n_samples
    grad_norm = utils.get_grad_norm(agent.actor.parameters())
    return {
        "actor/value": total_loss,
        "actor/grad_norm": grad_norm,
        # "actor/std": metrics.get("std")
    }


def critic(repay_buffer, agent, pi_optim, batch_size=32, gamma=0.99, epochs=10):
    # TODO n-step return here
    total_loss = torch.tensor(0.)
    n_batches = 0
    for _ in range(epochs):
        transition = repay_buffer.sample(batch_size)
        agent.zero_grad()
        loss = td_loss(agent, transition.state, transition.reward, transition.next_state, transition.done, gamma)
        loss.mean().backward()
        total_loss += loss.mean()
        # torch.nn.utils.clip_grad_value_(agent.critic.parameters(), 50.)
        n_batches += 1
        pi_optim.step()
    grad_norm = utils.get_grad_norm(agent.critic.parameters())
    total_loss = total_loss / n_batches
    return {
        "critic/td": total_loss.detach(),
        "critic/grad_norm": grad_norm.detach(),
    }


def td_loss(policy, s, r, s1, done, gamma):
    td = r + gamma * (1 - done) * policy.value(s1).squeeze() - policy.value(s).squeeze()
    loss = (0.5 * (td ** 2))
    return loss
