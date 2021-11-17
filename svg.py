import torch
from torch._vmap_internals import vmap

import torch.distributions as torch_dist
import config
import utils
from buffer import Transition

from dataclasses import replace


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
    action = mu + std * transition.noise
    reward = vmap(model.reward)(transition.state, action)
    next_state = vmap(model.dynamics)(transition.state, action)
    # TODO maybe missing IW if policy drifts off policy
    # verify_transition(transition, state, action, reward, next_state)
    return Transition(next_state, action, reward, next_state, done=transition.done, noise=transition.noise)


def _unroll(trajectory, agent, model, gamma):
    total_return = torch.zeros((1,))
    transition = trajectory[0]
    T = len(trajectory)
    for t, real_transition in enumerate(trajectory):
        real_transition = trajectory[t]
        transition = replace(real_transition, noise=real_transition.noise, done=real_transition.done,
                             next_state=transition.next_state)
        transition = Transition(*list(map(lambda x: x.unsqueeze(0), transition.__dict__.values())))
        transition = one_step(transition, agent, model)
        # Remember as the policy drift  the action drift and thus the visited state
        # Assume we get the reward upon transition to s1 and not at s1
        total_return = total_return + (gamma ** t) * transition.reward
    total_return += (1 - transition.done) * agent.value(transition.next_state).squeeze() * gamma ** T
    # we do not change the value function here, but we need to backprop throuhg
    # what happen if we replace the value function bootstrapped here with the one from pi^star?
    # this looks like the target of n-step td but off by one factor.
    # assume determintic action for the bootrap
    return total_return, transition


def _unroll_one_step(transition, agent, model, gamma):
    # this is only for simplicity
    transition = one_step(transition, agent, model)
    one_step_return = transition.reward + gamma * agent.value(transition.next_state)
    return one_step_return, transition


def unroll(trajectory, agent, model, gamma):
    if len(trajectory) == 1:
        # here we can use larger batch size. sampler does not support T x N
        return _unroll_one_step(trajectory[0], agent, model, gamma)
    else:
        return _unroll(trajectory, agent, model, gamma)


def actor_trajectory(replay_buffer, agent, model, pi_optim, horizon, gamma=0.99, epochs=1.):
    total_loss = torch.tensor(0.)
    for _ in range(epochs):
        pi_optim.zero_grad()
        n_samples = 0
        trajectory = replay_buffer.sample_partial(horizon)
        value, _ = unroll(trajectory, agent, model, gamma)
        (-value.mean()).backward()
        total_loss += value.mean().detach()
        n_samples += len(value)
        pi_optim.step()
        total_loss = total_loss.detach() / n_samples
    grad_norm = utils.get_grad_norm(agent.actor.parameters())
    return {
        "actor/value": total_loss,
        "actor/grad_norm": grad_norm.detach(),
        # "actor/std": metrics.get("std")
    }
def actor(replay_buffer, agent, model, pi_optim, batch_size=32, gamma=0.99, epochs=1.):
    total_loss = torch.tensor(0.)
    for _ in range(epochs):
        pi_optim.zero_grad()
        n_samples = 0
        for transition, _ in replay_buffer.sample(batch_size=batch_size):
            value, _ = unroll([transition], agent, model, gamma)
            (-value.mean()).backward()
            total_loss += value.mean().detach()
            n_samples += len(value)
        pi_optim.step()
        total_loss = total_loss.detach() / n_samples
    grad_norm = utils.get_grad_norm(agent.actor.parameters())
    return {
        "actor/value": total_loss,
        "actor/grad_norm": grad_norm.detach(),
        # "actor/std": metrics.get("std")
    }


def critic(repay_buffer, agent, pi_optim, batch_size=32, gamma=0.99, epochs=10):
    # TODO n-step return here
    total_loss = torch.tensor(0.)
    n_samples = 0
    for _ in range(epochs):
        for (s, a, r, s1, done, noise), target in repay_buffer.sample(batch_size):
            loss = 0.5 * (target - agent.value(s).squeeze()) ** 2
            agent.zero_grad()
            loss = td_loss(agent, s, r, s1, done, gamma)
            # loss = q_loss(agent.q_value, s, a, r, s1, a1, done, gamma)
            loss.mean().backward()
            total_loss += loss.mean()
            torch.nn.utils.clip_grad_value_(agent.critic.parameters(), 50.)
            n_samples += 1
            pi_optim.step()
    grad_norm = utils.get_grad_norm(agent.critic.parameters())

    total_loss = total_loss / n_samples
    return {
        "critic/td": total_loss.detach(),
        "critic/grad_norm": grad_norm.detach(),
    }



def q_loss(value, s, a, r, s1, a1, done, gamma):
    # sarsa style
    td = r + gamma * (1 - done) * value(s1, a1).squeeze() - value(s, a).squeeze()
    loss = (0.5 * (td ** 2))
    return loss


def td_loss(policy, s, r, s1, done, gamma):
    td = r + gamma * (1 - done) * policy.value(s1).squeeze() - policy.value(s).squeeze()
    loss = (0.5 * (td ** 2))
    return loss
