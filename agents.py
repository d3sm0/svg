import rlego
import torch
import torch.optim as optim
from torch._vmap_internals import vmap

import config
import utils


class SVGZero:
    def __init__(self, model, gamma=0.99):
        self.model = model
        self.gamma = gamma
        self.actor_optim = optim.SGD(model.actor.parameters(), lr=config.policy_lr)
        self.critic_optim = optim.SGD(
            [{"params": model.critic.parameters()},
             {"params": model.baseline.parameters()}]
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
        q_loss = (vtrace_target.q_estimate - q_tm1) * (1 - config.gamma) + utils.l2_norm(
            self.model.critic.named_parameters())
        td_loss = (vtrace_target.target_tm1 - v_tm1) * (1 - config.gamma)
        loss = 0.5 * q_loss.pow(2) + 0.5 * td_loss.pow(2)
        return loss.mean(), {"critic/td": td_loss.mean().detach(),
                             "critic/q_loss": q_loss.mean().detach()
                             }

    def _optimize_actor(self, batch):
        pi = self.model.actor(batch.state)
        new_actions = pi.rsample()
        reg = utils.l2_norm(self.model.actor.named_parameters())
        value = (1. - batch.done) * self.model.critic(batch.state, new_actions).squeeze(-1)
        return (-value).mean(), {"actor/value": value.mean().detach(),
                                 "actor/loc": pi.mean.mean(),
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
        optim.zero_grad()
        grad_norm = utils.get_grad_norm(model)
        loss, extra = loss_fn(transition)
        if not torch.isfinite(loss):
            raise ValueError("Loss is not finite.")
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model, 1.)
        grad_norm = utils.get_grad_norm(model)
        optim.step()
    # optim.zero_grad()
    return {
        f"{prefix}/loss": loss.detach(),
        f"{prefix}/grad_norm": grad_norm,
        **extra
    }


class SVG:
    def __init__(self, model, horizon=1, gamma=0.99):
        self.model = model
        from models import DynamicsLQR
        from envs.lqg import Lqg
        self.env = DynamicsLQR(Lqg())
        self.horizon = horizon
        self.gamma = gamma
        self.optim = torch.optim.SGD(self.model.parameters(), lr=config.model_lr)

    def _optimize_critic(self, batch):
        state = batch.state
        h = self.model.body(state)
        with torch.no_grad():
            v_t = self.model.critic(self.model.body(batch.next_state)).squeeze()
        v_tm1 = self.model.critic(h).squeeze()
        discount_t = (1 - batch.done) * self.gamma
        r_t = batch.reward
        vtrace_target = rlego.vtrace_td_error_and_advantage(v_tm1.detach(), v_t, r_t, discount_t,
                                                            rho_tm1=torch.ones_like(discount_t))
        td_loss = (vtrace_target.target_tm1 - v_tm1) * (1 - config.gamma)
        loss = 0.5 * td_loss.pow(2)
        return loss.mean(), {"critic/td": td_loss.mean().detach()}

    def optimize_actor(self, replay_buffer, epochs=1):
        actor_info = optimize_from_buffer(self.model.actor, self._optimize_actor, self.optim,
                                          replay_buffer, epochs, prefix="actor")
        return actor_info

    def optimize_critic(self, replay_buffer, epochs=1):
        value_info = optimize_from_buffer(self.model.critic, self._optimize_critic, self.optim, replay_buffer, epochs,
                                          prefix="critic")
        return value_info

    def _optimize_actor(self, batch):
        # Remember as the policy drift  the action drift and thus the visited state
        # Assume we get the reward upon transition to s1 and not at s1
        # we do not change the value function here, but we need to backprop throuhg
        # what happen if we replace the value function bootstrapped here with the one from pi^star?
        # this looks like the target of n-step td but off by one factor.
        # assume determintic action for the bootrap

        h = self.model.body(batch.state)
        transitions, pi = self._unroll(h, horizon=config.plan_horizon)

        rewards = [t[2] for t in transitions]
        s_tp1 = [t[3] for t in transitions]
        rewards = torch.stack(rewards, dim=1)
        # note terminal bias
        discount_t = torch.ones_like(rewards) * config.gamma
        v_t = self.model.critic(s_tp1[-1]).squeeze(dim=-1).detach()
        value = vmap(rlego.discounted_returns)(rewards, discount_t, v_t)
        return -value.sum(1).mean(), {"actor/value": value.mean().detach(),
                                      "actor/loc": pi.mean.mean(),
                                      # "actor/reg": reg.mean(),
                                      "actor/scale": pi.variance.mean()
                                      }

    def _optimize_model(self, batch):
        horizon = config.plan_horizon
        batch_size = config.batch_size
        batch_idx = torch.randint(0, len(batch.state) - horizon, size=(batch_size,))
        train_slice = []
        for idx in batch_idx:
            state_slice = batch.state[idx:idx + horizon]
            reward_slice = batch.reward[idx:idx + horizon]
            v_t = self.model.critic(self.model.body(batch.state[idx + horizon])).detach()
            target = rlego.discounted_returns(reward_slice, discount_t=torch.ones_like(reward_slice) * config.gamma,
                                              v_t=v_t) * (1 - config.gamma)
            train_slice.append((state_slice, reward_slice, target))
        total_loss = torch.tensor(0.)
        for (states, rewards, target) in train_slice:
            h = self.model.body(states[:1])
            transitions, pi = self._unroll(h, horizon=len(states))
            r_hat = torch.cat([t[2] for t in transitions])
            v_tm1 = torch.cat([t[-1] for t in transitions])
            reward_loss = (rewards - r_hat).pow(2).sum()
            value_loss = 0.5 * (target - v_tm1.squeeze(dim=-1)).pow(2).sum()
            total_loss = total_loss + (reward_loss + value_loss)
        total_loss = total_loss / batch_size
        return total_loss, {
            "model/model_loss": value_loss.detach(),
            # "model/actions": torch.linalg.norm(a_t[0]),
            # "model/states": torch.linalg.norm(s_tp1[0]),
            "model/reward_loss": reward_loss.detach(),
        }

    def optimize_model(self, replay_buffer, epochs=1):
        value_info = optimize_from_buffer(self.model, self._optimize_model, self.optim,
                                          replay_buffer, epochs, prefix="model")
        return value_info

    def _unroll(self, state, horizon):
        s_t = state
        transitions = []
        for t in range(horizon):
            pi = self.model.actor(s_t)
            a_t = pi.rsample()
            v_t = self.model.critic(s_t)
            s_tp1, r_t = self.model.dynamics(s_t, a_t)
            transitions.append((s_t, a_t, r_t, s_tp1, v_t))
        return transitions, pi
