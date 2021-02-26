import operator
from datetime import datetime
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.tensorboard as tb

from buffer import Buffer
from envs import pendulum_torch
from typing import NamedTuple


# TODO substitute with proper config object
config = SimpleNamespace(use_oracle=False, initial_steps=0,
                         policy_lr=3e-3, model_lr=1e-3,
                         train_on_buffer=False,
                         horizon=200,
                         max_steps=int(1e4),
                         envname="pendulum"
                         )


class Transition(NamedTuple):
    s: torch.tensor
    a: torch.tensor
    r: torch.tensor
    s1: torch.tensor
    done: torch.tensor


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

    def sample(self, batch_size, shuffle=False):
        idxs = torch.arange(self.__len__())
        if shuffle:
            idxs = idxs[torch.randperm(self.__len__())]

        idxs = torch.split(idxs, split_size_or_sections=batch_size)
        for idx in idxs:
            batch = operator.itemgetter(*idx)(self._data)
            s, a, r, s1, *_ = list(zip(*batch))
            s = torch.stack(s)
            a = torch.stack(a)
            s1 = torch.stack(s1)
            yield s, a, s1


class Policy(nn.Module):
    def __init__(self, obs_dim, action_dim, h_dim=32, mu_activation=None):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.mu_activation = mu_activation
        self.fc = nn.Sequential(*[nn.Linear(obs_dim, h_dim), nn.Tanh(), nn.Linear(h_dim, h_dim), nn.Tanh()])
        self.out = nn.Linear(h_dim, 2 * action_dim)

    def forward(self, s):
        h = self.fc(s)
        out = self.out(h)
        mu, sigma = torch.split(out, self.action_dim, -1)
        if self.mu_activation != None:
            mu = self.mu_activation(mu)
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
        # TODO @proecduralia this is a hack. It will break with new env
        self.r = env._r
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.fc = nn.Sequential(*[nn.Linear(obs_dim + action_dim, h_dim), nn.ELU()])
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
    s1_hat, sigma = dynamics.f(transition.s, transition.a)
    eta = (transition.s1 - s1_hat) / sigma
    next_state = s1_hat + sigma * eta.detach()

    a_hat, sigma = policy(transition.s)
    eps = (transition.a - a_hat) / sigma
    action = a_hat + sigma * eps.detach()
    return action, next_state


def unroll(dynamics, policy, traj, gamma=0.99):
    total_reward = 0
    state = traj[0][0]
    for t, transition in enumerate(traj):
        action, next_state = recreate_transition(transition, dynamics, policy)
        reward = dynamics.r(state, action)
        state = next_state
        total_reward += (gamma ** t) * reward
    return total_reward


def train(dynamics, policy, pi_optim, traj, opt_steps=1, grad_clip=5., gamma=0.99):
    total_reward = None
    true_norm = 0
    for _ in range(opt_steps):
        total_reward = unroll(dynamics, policy, traj, gamma=gamma)
        pi_optim.zero_grad()
        (-total_reward).backward()
        true_norm = get_grad_norm(parameters=policy.parameters())
        nn.utils.clip_grad_value_(policy.parameters(), grad_clip)
        pi_optim.step()
    return total_reward, true_norm


def generate_episode(env, policy, gamma=0.99):
    state = env.reset()
    trajectory = Trajectory()
    done = False
    total_reward = 0
    t = 0
    while not done:
        action, action_noise = policy.sample(state)
        next_state, reward, done, info = env.step(action)
        trajectory.append(Transition(state, action, reward, next_state, done))
        state = next_state
        total_reward += gamma ** t * reward
        t += 1
    return trajectory, total_reward


def train_model_on_traj(dynamics, traj, model_optim, batch_size=64):
    total_loss = 0.
    for (state, action, next_state) in traj.sample(batch_size=batch_size, shuffle=True):
        model_optim.zero_grad()
        mu, _ = dynamics(state, action)
        loss = nn.functional.mse_loss(mu, next_state)
        loss.backward()
        model_optim.step()
        total_loss += loss
    return total_loss/(len(traj)/batch_size)


def train_model_on_buffer(dynamics, buffer, model_optim, batch_size=64, n_batches=16):
    buffer_it = buffer.train_batches(batch_size)
    total_loss = 0
    for _ in range(n_batches):
        try:
            state, action, next_state = next(buffer_it)
        except StopIteration:
            break
        model_optim.zero_grad()
        mu, _ = dynamics(state, action)
        loss = nn.functional.mse_loss(mu, next_state)
        total_loss += loss
        loss.backward()
        model_optim.step()
    return total_loss / n_batches


from envs import torch_envs
from envs.zika_torch import ZikaEnv


def main():
    torch.manual_seed(0)
    if config.envname == 'zika':
        env = ZikaEnv()
        gamma = 0.5
        policy = Policy(env.observation_space.shape[0], env.action_space.shape[0],
                        h_dim=16, mu_activation=torch.sigmoid)
    elif config.envname == 'pendulum':
        env = torch_envs.Wrapper(pendulum_torch.Pendulum(), horizon=config.horizon)
        gamma = 0.99
        policy = Policy(env.observation_space.shape[0], env.action_space.shape[0], h_dim=4)

    if config.use_oracle:
        if config.envname != "pendulum":
            raise Exception("Oracle can only be used with pendulum")
        dynamics = RealDynamics(env)
    else:
        if config.train_on_buffer:
            buffer = Buffer(env.observation_space.shape[0], env.action_space.shape[0], 2048)
        dynamics = LearnedDynamics(env)

    pi_optim = optim.SGD(policy.parameters(), lr=config.policy_lr)
    model_optim = None if config.use_oracle else optim.SGD(dynamics.parameters(), lr=config.model_lr)

    dtm = datetime.now().strftime("%d-%H-%M-%S-%f")
    writer = tb.SummaryWriter(log_dir=f"logs/{dtm}")
    for epoch in range(int(config.max_steps)):
        traj, env_reward = generate_episode(env, policy, gamma=gamma)
        writer.add_scalar("env/return", env_reward, global_step=epoch)
        writer.add_scalar("env/action", float(torch.mean(traj[0].a)), global_step=epoch)
        if not config.use_oracle:
            if config.train_on_buffer:
                buffer.add_trajectory(traj)
                model_loss = train_model_on_buffer(dynamics, buffer, model_optim)
            else:
                model_loss = train_model_on_traj(dynamics, traj, model_optim)
            writer.add_scalar("train/model_loss", model_loss, global_step=epoch)

        if epoch > config.initial_steps or config.use_oracle:
            ret, grad_norm = train(dynamics, policy, pi_optim, traj, gamma=gamma)
            writer.add_scalar("train/return", ret, global_step=epoch)
            writer.add_scalar("train/grad_norm", grad_norm, global_step=epoch)


if __name__ == '__main__':
    main()
