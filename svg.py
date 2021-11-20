import torch

import config
import utils


def actor(replay_buffer, agent, pi_optim, batch_size=32, epochs=1):
    total_loss = torch.tensor(0.)
    for _ in range(epochs):
        transition = replay_buffer.sample(batch_size=batch_size)
        pi_optim.zero_grad()
        # this backprpop via the policy
        value = agent.value(transition.state, agent.rsample(transition.state))
        (-value.mean()).backward()
        torch.nn.utils.clip_grad_value_(agent.actor.parameters(), config.grad_clip)
        pi_optim.step()
        total_loss += value.mean().detach()
    total_loss = total_loss / epochs
    grad_norm = utils.get_grad_norm(agent.actor.parameters())
    return {
        "actor/value": total_loss,
        "actor/grad_norm": grad_norm,
    }


def critic(repay_buffer, agent, pi_optim, batch_size=32, gamma=0.99, epochs=10):
    # TODO n-step return here
    total_loss = torch.tensor(0.)
    for _ in range(epochs):
        transition = repay_buffer.sample(batch_size)
        agent.zero_grad()
        # this is quite incorrect as is not the same action but the one from a deterministcit policy
        next_action, _ = agent.get_action(transition.next_state)
        loss = q_loss(agent, transition.state, transition.action, transition.reward, transition.next_state,
                      next_action, transition.done, gamma)
        loss.mean().backward()
        torch.nn.utils.clip_grad_value_(agent.critic.parameters(), config.grad_clip)
        total_loss += loss.mean()
        pi_optim.step()
    grad_norm = utils.get_grad_norm(agent.critic.parameters())
    total_loss = total_loss / epochs
    return {
        "critic/td": total_loss.detach(),
        "critic/grad_norm": grad_norm.detach(),
    }


def q_loss(agent, s, a, r, s1, a1, done, gamma):
    td = r + gamma * (1 - done) * agent.target_value(s1, a1).squeeze().detach() - agent.value(s, a).squeeze()
    loss = (0.5 * (td ** 2))
    return loss
