import torch

from models import polyak_update
from svg import unroll_trajectory, td_loss, one_step, q_loss


class SVG:
    horizon: int = 2

    def __init__(self, agent, horizon: int = 2):
        self.agent = agent
        self.horizon = horizon

    def get_action(self, s):
        with torch.no_grad():
            mu, sigma = self.actor.forward(s)
        eps = torch.randn(size=mu.shape).to(mu.device)
        return mu + sigma * eps, eps

    @property
    def actor(self):
        return self.agent.actor

    @property
    def critic(self):
        return self.agent.critic

    def get_value_gradient(self, trajectory, dynamics, gamma):
        assert len(trajectory) == self.horizon
        total_return, metrics = unroll_trajectory(trajectory, self.agent, dynamics, gamma)
        return total_return, metrics

    def get_value_loss(self, batch, gamma):
        v_t = self.critic(batch.next_state)
        v_tm1 = self.critic(batch.state)
        loss = td_loss(batch.reward, v_tm1, v_t, batch.done, gamma)
        return 0.5 * (loss ** 2)

    def update_target(self, tau):
        return


class SVGOne(SVG):
    def __init__(self, agent):
        super(SVGOne, self).__init__(agent, horizon=1)

    def get_value_gradient(self, transitions, dynamics, gamma):
        transition, metrics = one_step(transitions, self.agent, dynamics)
        one_step_return = transition.reward + gamma * (1 - transition.done) * self.agent.value(
            transition.next_state).squeeze()
        return one_step_return, metrics


class SVGZero(SVG):
    def __init__(self, agent, horizon: int = 0):
        super(SVGZero, self).__init__(agent, horizon=0)

    @property
    def target_critic(self):
        return self.agent.target_critic

    def get_value_gradient(self, transitions, dynamics, gamma):
        value = (1 - transitions.done) * self.value(transitions.state, self.rsample(transitions.state)).squeeze()
        return value, {"duration": 0, "value": value.detach()}

    def value(self, state, action):
        state_action = torch.cat([state, action], -1)
        return self.agent.critic(state_action)

    def target_value(self, state, action):
        state_action = torch.cat([state, action], -1)
        return self.agent.target_critic(state_action)

    def get_value_loss(self, batch, gamma):
        a_t, _ = self.get_action(batch.next_state)
        q_t = self.target_value(batch.next_state, a_t).squeeze()
        q_tm1 = self.value(batch.state, batch.action).squeeze()
        return 0.5 * q_loss(batch.reward, q_tm1, q_t, batch.done, gamma) ** 2

    def rsample(self, state):
        mu, std = self.actor(state)
        eps = torch.randn(size=mu.shape)
        return mu + std * eps.detach()

    def update_target(self, tau=1.):
        polyak_update(self.critic.parameters(), self.target_critic.parameters(), tau)
