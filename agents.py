import rlego
import torch
import torch.optim as optim

import config
import utils
from svg import unroll_trajectory, one_step


class SVG:
    horizon: int = 2

    def __init__(self, agent, horizon: int = 2):
        self.agent = agent
        self.horizon = horizon

    def get_value_gradient(self, trajectory, dynamics, gamma):
        assert len(trajectory) == self.horizon
        total_return, metrics = unroll_trajectory(trajectory, self.agent, dynamics, gamma)
        return total_return, metrics

    def get_value_loss(self, batch, gamma):
        v_t = self.critic(batch.next_state)
        v_tm1 = self.critic(batch.state)
        discount_t = (1 - batch.done) * gamma
        loss = rlego.td_learning(v_tm1, batch.reward, discount_t, v_t, stop_grad=True)
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


class SVGZero:
    def __init__(self, agent, gamma=0.99):
        self.model = agent
        self.gamma = gamma
        self.actor_optim = optim.SGD(agent.actor.parameters(), lr=config.policy_lr)
        self.critic_optim = optim.SGD(
            [{"params": agent.critic.parameters()},
             {"params": agent.baseline.parameters()}]
            , lr=config.critic_lr)

    def update_target(self, tau=1.):
        rlego.polyak_update(self.model.baseline.parameters(), self.model.target_critic.parameters(), tau)

    def _optimize_critic(self, batch):
        with torch.no_grad():
            v_t = self.model.baseline(batch.next_state).squeeze()
        v_tm1 = self.model.baseline(batch.state).squeeze()
        q_tm1 = self.model.critic(batch.state, batch.action).squeeze()
        discount_t = (1 - batch.done) * self.gamma
        r_t = batch.reward
        vtrace_target = rlego.vtrace_td_error_and_advantage(v_tm1.detach(), v_t, r_t, discount_t,
                                                            rho_tm1=torch.ones_like(discount_t))
        q_loss = (vtrace_target.q_estimate - q_tm1) * (1-config.gamma) + utils.l2_norm(self.model.critic.named_parameters())
        td_loss = (vtrace_target.target_tm1 - v_tm1) * (1-config.gamma)
        loss = 0.5 * q_loss.pow(2) + 0.5 * td_loss.pow(2)
        return loss.mean(), {"critic/td": td_loss.mean().detach(),
                             "critic/q_loss": q_loss.mean().detach()
                             }

    def _optimize_actor(self, batch):
        pi = self.model.actor(batch.state)
        new_actions = pi.rsample()
        reg = utils.l2_norm(self.model.actor.named_parameters())
        value = (1. - batch.done) * self.model.critic(batch.state, new_actions).squeeze(-1)
        return (-value).mean(), {"actor/value": value.mean().detach(), "actor/loc": pi.mean.mean(),
                                       "actor/reg": reg.mean(),
                                       "actor/scale": pi.variance.mean()}

    def optimize_critic(self, replay_buffer, epochs=1):
        value_info = optimize_from_buffer(self.model.critic, self._optimize_critic, self.critic_optim,
                                          replay_buffer, epochs, prefix="critic")
        return value_info

    def optimize_actor(self, replay_buffer, epochs=1):
        actor_info = optimize_from_buffer(self.model.actor, self._optimize_actor, self.actor_optim,
                                          replay_buffer, epochs, prefix="actor")
        return actor_info


def optimize_from_buffer(model, loss_fn, optim, repay_buffer, epochs=1, prefix=""):
    loss = torch.tensor(0.)
    grad_norm = torch.tensor(0.)
    extra = {}
    for _ in range(epochs):
        transition = repay_buffer.transpose()
        model.zero_grad()
        loss, extra = loss_fn(transition)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model, 1.)
        grad_norm = utils.get_grad_norm(model)
        optim.step()
    return {
        f"{prefix}/loss": loss.detach(),
        f"{prefix}/grad_norm": grad_norm,
        **extra
    }
