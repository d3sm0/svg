from brax.envs import to_torch, create_gym_env
import torch

env = create_gym_env("inverted_pendulum")
env = to_torch.JaxToTorchWrapper(env)

import ipdb; ipdb.set_trace() # yapf: disable TODO slog
state =env.reset()
