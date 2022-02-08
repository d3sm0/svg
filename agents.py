import rlego
import torch
import torch.optim as optim

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
        loss, extra = loss_fn(transition)
        if not torch.isfinite(loss):
            raise ValueError("Loss is not finite.")
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 5.)
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
        self.optim = torch.optim.Adam(
            [
                {"params": self.model.actor.parameters()},
                {"params": self.model.critic.parameters()},
                {"params": self.model.q.parameters()},
                {"params": self.model.body.parameters()},
                {"params": self.model.dynamics.parameters()},
            ],
            lr=config.model_lr)
        self.planner_optim = torch.optim.SGD([{"params": self.model.planner.parameters()}], lr=config.policy_lr)

    def plan(self, state):
        h = self.model.body(state)
        transitions, pi = self._unroll(h, horizon=1, actor=self.model.planner)

        rewards = [t[2] for t in transitions]
        s_tp1 = [t[3] for t in transitions]
        rewards = torch.stack(rewards)

        discount_t = torch.ones_like(rewards) * config.gamma
        v_t = self.model.critic(s_tp1[-1]).squeeze(dim=-1).detach()
        value = rlego.discounted_returns(rewards, discount_t, v_t) * (1 - config.gamma)
        self.planner_optim.zero_grad()
        (-value.sum()).backward()
        self.optim.zero_grad()
        self.planner_optim.step()
        with torch.no_grad():
            pi = self.model.planner(h)
        return pi

    def optimize_model(self, replay_buffer, epochs=1):
        value_info = optimize_from_buffer(self.model, self._optimize_model, self.optim,
                                          replay_buffer, epochs, prefix="model")
        return value_info

    def _optimize_model(self, batch):
        horizon = config.plan_horizon
        batch_size = config.batch_size
        batch_idx = torch.randint(0, len(batch.state) - horizon, size=(batch_size,))
        train_slice = []
        for idx in batch_idx:
            state_slice = batch.state[idx:idx + horizon]
            reward_slice = batch.reward[idx:idx + horizon]
            loc, scale = torch.split(torch.stack(batch.info[idx:idx + horizon]), dim=-1, split_size_or_sections=1)
            pi_env = torch.distributions.Normal(loc, scale * config.gamma)
            v_t = self.model.critic(self.model.body(batch.state[idx + horizon])).detach()
            train_slice.append((state_slice, reward_slice, v_t, pi_env))

        total_loss = torch.tensor(0.)
        for (states, rewards, v_t, pi_env) in train_slice:
            h = self.model.body(states[:1])
            transitions, pi = self._unroll(h, horizon=len(states), actor=self.model.actor)
            r_hat = torch.cat([t[2] for t in transitions])
            v_tm1 = torch.cat([t[-2] for t in transitions]).squeeze(dim=-1)
            q_tm1 = torch.cat([t[-1] for t in transitions]).squeeze(dim=-1)
            a_tm1 = torch.cat([t[1] for t in transitions])

            pi_true = torch.stack([t[-3] for t in transitions]).squeeze(dim=-1)
            pi_true = torch.distributions.Normal(*torch.split(pi_true, split_size_or_sections=1, dim=-1))
            pi_loss = torch.distributions.kl_divergence(pi_true, pi_env).sum(dim=-1).sum()
            discount_t = torch.ones_like(r_hat) * config.gamma
            rho_tm1 = torch.exp(pi.log_prob(a_tm1) - pi_env.log_prob(a_tm1)).sum(dim=-1).exp().detach()
            v_trace_output = rlego.vtrace_td_error_and_advantage(v_tm1.squeeze(dim=-1), v_t, rewards, discount_t,
                                                                 rho_tm1)

            target = v_trace_output.target_tm1 * (1 - config.gamma)
            q_target = v_trace_output.q_estimate * (1 - config.gamma)
            reward_loss = (rewards.detach() - r_hat).pow(2).sum()
            value_loss = 0.5 * (target.detach() - v_tm1).pow(2).sum()
            q_loss = 0.5 * (q_target.detach() - q_tm1).pow(2).sum()
            total_loss = total_loss + (value_loss + reward_loss + pi_loss + q_loss )
        total_loss = total_loss / batch_size
        return total_loss, {
            "model/model_loss": value_loss.detach(),
            "model/q_loss": q_loss.detach(),
            "model/pi": pi_loss.detach(),
            "actor/rho": rho_tm1.mean(),
            "actor/loc": pi.mean.mean(),
            "actor/scale": pi.variance.mean(),
            # "model/actions": torch.linalg.norm(a_t[0]),
            # "model/states": torch.linalg.norm(s_tp1[0]),
            "model/reward_loss": reward_loss.detach(),
        }

    def _unroll(self, state, horizon, actor):
        s_t = state
        transitions = []
        for t in range(horizon):
            pi = actor(s_t)
            a_t = pi.rsample()
            q_t = self.model.q(s_t, a_t)
            v_t = self.model.critic(s_t)
            s_tp1, r_t = self.model.dynamics(s_t, a_t)
            transitions.append((s_t, a_t, r_t, s_tp1, torch.cat([pi.loc, pi.scale]), v_t, q_t))
            s_t = s_tp1
        return transitions, pi

    def update_target(self, tau=1.):
        rlego.polyak_update(self.model.actor.parameters(), self.model.planner.parameters(), tau)
