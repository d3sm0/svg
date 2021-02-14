import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.tensorboard as tb

from envs import pendulum_torch


# TODO recall that gradient explode

class Policy(nn.Module):
    def __init__(self, obs_space, action_space, h_dim=32):
        super(Policy, self).__init__()
        self.fc = nn.Sequential(*[nn.Linear(obs_space, h_dim), nn.Tanh(), nn.Linear(h_dim, h_dim), nn.Tanh()])
        self.out = nn.Linear(h_dim, 2 * action_space)

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


def get_grad_norm(parameters):
    return torch.norm(torch.cat([p.grad.flatten() for p in parameters]))


def infer_noise(transition, dynamics, policy):
    s, a, r, s1, d, eps, eta = transition

    s1_hat, sigma = dynamics.f(s, a)
    eta = (s1 - s1_hat) / sigma

    a_hat, sigma = policy(s)
    eps = (a - a_hat) / sigma
    return eps.detach(), eta.detach()


def unroll(dynamics, policy, traj):
    gamma = 0.99
    total_reward = 0
    s = traj[0][0]
    # scale = 0.5 ** 2
    for t, transition in enumerate(traj):
        action_noise, model_noise = infer_noise(transition, dynamics, policy)
        mu, sigma = policy(s)
        a = mu + sigma * action_noise
        r = dynamics.r(s, a)
        s1 = dynamics.f(s, a)
        mu, sigma = dynamics.f(s, a)
        s1 = mu + sigma * model_noise
        s = s1
        total_reward += (gamma ** t) * r
        # s = s * scale + (1 - scale) * s.detach()
        # total_reward =  total_reward * scale + (1-scale) * total_reward.detach()
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


class Trajectory:

    def __init__(self):
        self._data = []

    def __getitem__(self, item):
        return self._data[item]

    def append(self, transition):
        self._data.append(transition)


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


def main():
    torch.manual_seed(0)
    env = pendulum_torch.Pendulum()
    policy = Policy(3, 1)
    dynamics = RealDynamics(env)
    dtm = datetime.now().strftime("%d-%H-%M-%S-%f")
    pi_optim = optim.SGD(policy.parameters(), lr=1e-3)
    writer = tb.SummaryWriter(log_dir=f"logs/{dtm}")
    save_every = int(1e3)
    for epoch in range(int(1e4)):
        traj, env_reward = generate_episode(env, policy)
        reward, grad_norm = train(dynamics, policy, pi_optim, traj)
        if epoch % save_every == 0:
            torch.save(policy.state_dict(), os.path.join("logs", dtm, "ckp.pb"))
        writer.add_scalar("train/reward", reward, global_step=epoch)
        writer.add_scalar("env/reward", env_reward, global_step=epoch)
        writer.add_scalar("train/grad_norm", grad_norm, global_step=epoch)


if __name__ == '__main__':
    main()
