import operator
from typing import NamedTuple

import torch
import torch.nn as nn


def l2_norm(parameters):
    grad_norm = 0
    for n, p in parameters:
        if "bias" in n:
            continue
        grad_norm += p.norm().cpu()
    return grad_norm


def get_grad_norm(parameters):
    grad_norm = torch.tensor(0.)
    for p in parameters:
        if p.grad is not None:
            grad_norm += p.grad.norm().cpu()
    return grad_norm


def lambda_returns(r_t: torch.Tensor, discount_t: torch.Tensor, v_t: torch.Tensor, lambda_: float = 1.,
                   stop_target_gradients: bool = False) -> torch.Tensor:
    """Estimates a multistep truncated lambda return from a trajectory.
    Given a a trajectory of length `T+1`, generated under some policy π, for each
    time-step `t` we can estimate a target return `G_t`, by combining rewards,
    discounts, and state values, according to a mixing parameter `lambda`.
    The parameter `lambda_`  mixes the different multi-step bootstrapped returns,
    corresponding to accumulating `k` rewards and then bootstrapping using `v_t`.
      rₜ₊₁ + γₜ₊₁ vₜ₊₁
      rₜ₊₁ + γₜ₊₁ rₜ₊₂ + γₜ₊₁ γₜ₊₂ vₜ₊₂
      rₜ₊₁ + γₜ₊₁ rₜ₊₂ + γₜ₊₁ γₜ₊₂ rₜ₊₂ + γₜ₊₁ γₜ₊₂ γₜ₊₃ vₜ₊₃
    The returns are computed recursively, from `G_{T-1}` to `G_0`, according to:
      Gₜ = rₜ₊₁ + γₜ₊₁ [(1 - λₜ₊₁) vₜ₊₁ + λₜ₊₁ Gₜ₊₁].
    In the `on-policy` case, we estimate a return target `G_t` for the same
    policy π that was used to generate the trajectory. In this setting the
    parameter `lambda_` is typically a fixed scalar factor. Depending
    on how values `v_t` are computed, this function can be used to construct
    targets for different multistep reinforcement learning updates:
      TD(λ):  `v_t` contains the state value estimates for each state under π.
      Q(λ):  `v_t = max(q_t, axis=-1)`, where `q_t` estimates the action values.
      Sarsa(λ):  `v_t = q_t[..., a_t]`, where `q_t` estimates the action values.
    In the `off-policy` case, the mixing factor is a function of state, and
    different definitions of `lambda` implement different off-policy corrections:
      Per-decision importance sampling:  λₜ = λ ρₜ = λ [π(aₜ|sₜ) / μ(aₜ|sₜ)]
      V-trace, as instantiated in IMPALA:  λₜ = min(1, ρₜ)
    Note that the second option is equivalent to applying per-decision importance
    sampling, but using an adaptive λ(ρₜ) = min(1/ρₜ, 1), such that the effective
    bootstrap parameter at time t becomes λₜ = λ(ρₜ) * ρₜ = min(1, ρₜ).
    This is the interpretation used in the ABQ(ζ) algorithm (Mahmood 2017).
    Of course this can be augmented to include an additional factor λ.  For
    instance we could use V-trace with a fixed additional parameter λ = 0.9, by
    setting λₜ = 0.9 * min(1, ρₜ) or, alternatively (but not equivalently),
    λₜ = min(0.9, ρₜ).
    Estimated return are then often used to define a td error, e.g.:  ρₜ(Gₜ - vₜ).
    See "Reinforcement Learning: An Introduction" by Sutton and Barto.
    (http://incompleteideas.net/sutton/book/ebook/node74.html).
    Args:
      r_t: sequence of rewards rₜ for timesteps t in [1, T].
      discount_t: sequence of discounts γₜ for timesteps t in [1, T].
      v_t: sequence of state values estimates under π for timesteps t in [1, T].
      lambda_: mixing parameter; a scalar or a vector for timesteps t in [1, T].
      stop_target_gradients: bool indicating whether or not to apply stop gradient
        to targets.
    Returns:
      Multistep lambda returns.
    """
    # assert_rank([r_t, discount_t, v_t, lambda_], [1, 1, 1, {0, 1}])
    # assert_type([r_t, discount_t, v_t, lambda_], float)
    # assert_equal_shape([r_t, discount_t, v_t])

    # If scalar make into vector.
    lambda_ = torch.ones_like(discount_t) * lambda_

    # Work backwards to compute `G_{T-1}`, ..., `G_0`.
    returns = []
    g = v_t[-1]
    for i in reversed(range(v_t.shape[0])):
        g = r_t[i] + discount_t[i] * ((1 - lambda_[i]) * v_t[i] + lambda_[i] * g)
        returns.insert(0, g)

    returns = torch.tensor(returns)
    if stop_target_gradients:
        returns.detach_()

    return returns


def n_step_bootstrapped_returns(r_t: torch.Tensor, discount_t: torch.Tensor, v_t: torch.Tensor, n: int, lambda_t: float = 1.,
                                stop_target_gradients: bool = False) -> torch.Tensor:
    """Computes strided n-step bootstrapped return targets over a sequence.
    The returns are computed according to the below equation iterated `n` times:
       Gₜ = rₜ₊₁ + γₜ₊₁ [(1 - λₜ₊₁) vₜ₊₁ + λₜ₊₁ Gₜ₊₁].
    When lambda_t == 1. (default), this reduces to
       Gₜ = rₜ₊₁ + γₜ₊₁ * (rₜ₊₂ + γₜ₊₂ * (... * (rₜ₊ₙ + γₜ₊ₙ * vₜ₊ₙ ))).
    Args:
      r_t: rewards at times [1, ..., T].
      discount_t: discounts at times [1, ..., T].
      v_t: state or state-action values to bootstrap from at time [1, ...., T].
      n: number of steps over which to accumulate reward before bootstrapping.
      lambda_t: lambdas at times [1, ..., T]. Shape is [], or [T-1].
      stop_target_gradients: bool indicating whether or not to apply stop gradient
        to targets.
    Returns:
      estimated bootstrapped returns at times [0, ...., T-1]
    """
    # chex.assert_rank([r_t, discount_t, v_t, lambda_t], [1, 1, 1, {0, 1}])
    # chex.assert_type([r_t, discount_t, v_t, lambda_t], float)
    # chex.assert_equal_shape([r_t, discount_t, v_t])
    seq_len = r_t.shape[0]

    # Maybe change scalar lambda to an array.
    lambda_t = torch.ones_like(discount_t) * lambda_t

    # Shift bootstrap values by n and pad end of sequence with last value v_t[-1].
    pad_size = min(n - 1, seq_len)
    targets = torch.cat([v_t[n - 1:], torch.tensor([v_t[-1]] * pad_size)])

    # Pad sequences. Shape is now (T + n - 1,).
    r_t = torch.cat([r_t, torch.zeros(n - 1)])
    discount_t = torch.cat([discount_t, torch.ones(n - 1)])
    lambda_t = torch.cat([lambda_t, torch.ones(n - 1)])
    v_t = torch.cat([v_t, torch.tensor([v_t[-1]] * (n - 1))])

    # Work backwards to compute n-step returns.
    for i in reversed(range(n)):
        r_ = r_t[i:i + seq_len]
        discount_ = discount_t[i:i + seq_len]
        lambda_ = lambda_t[i:i + seq_len]
        v_ = v_t[i:i + seq_len]
        targets = r_ + discount_ * ((1. - lambda_) * v_ + lambda_ * targets)

    if stop_target_gradients:
        targets.detach_()

    return targets
