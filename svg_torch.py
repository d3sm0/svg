from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.tensorboard as tb
from buffer import Buffer

from envs import pendulum_torch


# TODO substitute with proper config object
from types import SimpleNamespace
config = SimpleNamespace(use_oracle=False, initial_steps=100,
                         policy_lr=2e-2, model_lr=1e-3,
                         train_on_buffer=False)


class Trajectory:
    def __init__(self):
        self._data = []

    def __getitem__(self, item):
        return self._data[item]

    def append(self, transition):
        self._data.append(transition)

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)


class Policy(nn.Module):
    def __init__(self, obs_dim, action_dim, h_dim=32):
        super(Policy, self).__init__()
        self.fc = nn.Sequential(*[nn.Linear(obs_dim, h_dim), nn.Tanh(), nn.Linear(h_dim, h_dim), nn.Tanh()])
        self.out = nn.Linear(h_dim, 2 * action_dim)

    def forward(self, s):
        h = self.fc(s)
        out = self.out(h)
        mu, sigma = torch.split(out, 1, -1)  # TODO wierd stuf
        sigma = F.softplus(sigma)
        return mu, sigma

    @torch.no_grad()
    def sample(self, s):
        mu, sigma = self(s)
        eps = torch.randn(mu.shape)
        a = (mu + sigma * eps)
        return a, eps

    def rsample(self, s):
        mu, sigma = self(s)
        eps = torch.randn(mu.shape)
        a = (mu + sigma * eps)
        return a, eps


class RealDynamics:
    def __init__(self, env):
        self._f = env._f
        self.r = env._r
        self.std = 1

    def f(self, s, a):
        mu, sigma = self._f(s, a), self.std
        return mu, sigma


class LearnedDynamics(nn.Module):
    def __init__(self, env, h_dim=32):
        super().__init__()
        self.r = env._r
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.fc = nn.Sequential(*[nn.Linear(obs_dim+action_dim, h_dim), nn.ELU()])
        self.out = nn.Linear(h_dim, obs_dim)
        self.std = 1

    def forward(self, s, a):
        h = self.fc(torch.cat((s, a), dim=-1))
        mu = self.out(h) + s
        return mu, self.std

    def f(self, s, a):
        return self(s, a)


def get_grad_norm(parameters):
    return torch.norm(torch.cat([p.grad.flatten() for p in parameters]))


def recreate_transition(transition, dynamics, policy):
    s, a, r, s1, d, eps, eta = transition

    s1_hat, sigma = dynamics.f(s, a)
    eta = (s1 - s1_hat) / sigma
    mu, sigma = dynamics.f(s, a)
    ns = mu + sigma * eta.detach()

    a_hat, sigma = policy(s)
    eps = (a - a_hat) / sigma
    mu, sigma = policy(s)
    ac = mu + sigma * eps.detach()
    return ac, ns


def unroll(dynamics, policy, traj):
    gamma = 0.99
    total_reward = 0
    s = traj[0][0]
    for t, transition in enumerate(traj):
        a, ns = recreate_transition(transition, dynamics, policy)
        r = dynamics.r(s, a)
        s = ns
        total_reward += (gamma ** t) * r
    return total_reward


def train(dynamics, policy, pi_optim, traj, opt_steps=1):
    for _ in range(opt_steps):
        total_reward = unroll(dynamics, policy, traj)
        pi_optim.zero_grad()
        (-total_reward).backward()
        true_norm = get_grad_norm(parameters=policy.parameters())
        nn.utils.clip_grad_value_(policy.parameters(), 5.)
        pi_optim.step()
    return total_reward, true_norm


def generate_episode(env, policy, gamma=0.99):
    s = env.reset()
    trajectory = Trajectory()
    d = False
    total_reward = 0
    t = 0
    while not d:
        a, eps = policy.sample(s)
        s1, r, d, info = env.step(a)
        trajectory.append((s, a, r, s1, d, eps, 0.))
        s = s1
        total_reward += gamma ** t * r
        t += 1
    return trajectory, total_reward


def train_model_on_traj(dynamics, traj, model_optim, batch_size=64):
    idxs = torch.randperm(len(traj))
    states = torch.stack([transition[0] for transition in traj])[idxs]
    actions = torch.stack([transition[1] for transition in traj])[idxs]
    next_states = torch.stack([transition[3] for transition in traj])[idxs]

    for s, a, ns in zip(torch.split(states, batch_size), torch.split(actions, batch_size), torch.split(next_states, batch_size)):
        model_optim.zero_grad()
        mu, _ = dynamics(s, a)
        loss = nn.functional.mse_loss(mu, ns)
        loss.backward()
        model_optim.step()
    return loss


def train_model_on_buffer(dynamics, buffer, model_optim, batch_size=64, n_batches=16):
    buffer_it = buffer.train_batches(batch_size)

    for i in range(n_batches):
        try:
            s, a, ns = next(buffer_it)
        except StopIteration:
            break
        model_optim.zero_grad()
        mu, _ = dynamics(s, a)
        loss = nn.functional.mse_loss(mu, ns)
        loss.backward()
        model_optim.step()
    return loss


def main():
    torch.manual_seed(0)
    env = pendulum_torch.Pendulum()
    policy = Policy(3, 1)
    if config.use_oracle:
        dynamics = RealDynamics(env)
    else:
        if config.train_on_buffer:
            buffer = Buffer(3, 1, 2048)
        dynamics = LearnedDynamics(env)

    pi_optim = optim.SGD(policy.parameters(), lr=config.policy_lr)
    model_optim = None if config.use_oracle else optim.SGD(dynamics.parameters(), lr=config.model_lr)

    dtm = datetime.now().strftime("%d-%H-%M-%S-%f")
    writer = tb.SummaryWriter(log_dir=f"logs/{dtm}")
    for epoch in range(int(1e4)):
        traj, env_reward = generate_episode(env, policy)
        writer.add_scalar("env/return", env_reward, global_step=epoch)
        writer.add_scalar("env/action", float(traj[0][1]), global_step=epoch)
        if not config.use_oracle:
            if config.train_on_buffer:
                buffer.add_trajectory(traj)
                model_loss = train_model_on_buffer(dynamics, buffer, model_optim)
            else:
                model_loss = train_model_on_traj(dynamics, traj, model_optim)
            writer.add_scalar("train/model_loss", model_loss, global_step=epoch)

        if epoch > config.initial_steps or config.use_oracle:
            ret, grad_norm = train(dynamics, policy, pi_optim, traj)
            writer.add_scalar("train/return", ret, global_step=epoch)
            writer.add_scalar("train/grad_norm", grad_norm, global_step=epoch)


if __name__ == '__main__':
    main()
