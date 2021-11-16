import torch
from torch._vmap_internals import vmap

import torch.distributions as torch_dist
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
    varianve = []
    for t, transition in enumerate(trajectory):
        # Remember as the policy drift  the action drift and thus the visited state
        mu, std = agent.forward(state)
        varianve.append(std)
        action = mu + std * transition.noise
        next_state = model.dynamics(state, action)
        reward = model.reward(state, action)
        # verify_transition(transition, state, action, reward, next_state)
        # action, next_state = recreate_transition(transition, policy)
        # Assume we get the reward upon transition to s1 and not at s1
        total_return += (gamma ** t) * reward
        state = next_state
    # we do not change the value function here, but we need to backprop throuhg
    # what happen if we replace the value function bootstrapped here with the one from pi^star?
    # this looks like the target of n-step td but off by one factor.
    total_return += (1 - transition.done) * agent.value(state).squeeze() * gamma ** config.horizon
    return total_return, { "std": torch.cat(varianve).mean()
    }


def one_step(transition, agent, model, gamma):
    mu, std = agent.forward(transition.state)
    action = mu + std * transition.noise
    r = vmap(model.reward)(transition.state, action)
    next_state = vmap(model.dynamics)(transition.state, action)
    # TODO maybe missing IW if policy drifts off policy
    value = r + gamma * (1 - transition.done) * agent.value(next_state).squeeze()
    return value, {
        "std": torch.linalg.norm(std, 1).mean()
    }


def actor_trajectory(trajectory, agent, model, pi_optim, gamma=0.99, horizon=10):
    trajectory_partial = trajectory.sample_partial(horizon)
    # The policy needs to move slower than the critic
    # otherwise the truncation breaks and future value estimates
    # are gone
    for _ in range(5):
        pi_optim.zero_grad()
        value, metrics = unroll(trajectory_partial, agent, model, gamma)
        (-value).backward()
        grad_norm = utils.get_grad_norm(agent.actor.parameters())
    # torch.nn.utils.clip_grad_value_(agent.actor.parameters(), config.grad_clip)
    pi_optim.step()
    return {
        "actor/value": value.detach(),
        "actor/grad_norm": grad_norm.detach(),
        "actor/std": metrics.get("std")
    }


def actor(replay_buffer, agent, model, pi_optim, batch_size=32, gamma=0.99,epochs=1.):
    total_loss = torch.tensor(0.)
    n_samples = 0
    for _ in range(epochs):
        for transition, _ in replay_buffer.sample(batch_size=batch_size):
            value, metrics = one_step(transition, agent, model, gamma=gamma)
            pi_optim.zero_grad()
            (-value).mean().backward()
            pi_optim.step()
            total_loss += value.mean().detach()
            n_samples += 1
    grad_norm = utils.get_grad_norm(agent.actor.parameters())
    return {
        "actor/value": total_loss.detach() / n_samples,
        "actor/grad_norm": grad_norm.detach(),
        "actor/std": metrics.get("std")
    }


def critic(repay_buffer, agent, pi_optim, batch_size=32, gamma=0.99, epochs=10):
    # TODO n-step return here
    total_loss = torch.tensor(0.)
    n_samples = 0
    for _ in range(epochs):
        for (s, a, r, s1, done, noise), target in repay_buffer.sample(batch_size):
            # loss = 0.5 * (target - agent.value(s).squeeze())**2
            agent.zero_grad()
            loss = td_loss(agent, s, r, s1, done, gamma)
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


def td_loss(policy, s, r, s1, done, gamma):
    td = r + gamma * (1 - done) * policy.value(s1).squeeze() - policy.value(s).squeeze()
    loss = (0.5 * (td ** 2))
    return loss
