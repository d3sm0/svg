import sys

import torch

DEBUG = sys.gettrace() is not None
env_id = "inverted_pendulum"
should_render = False
proc_num = 1
host = ""
sweep_yaml = ""  # k"sweep.yaml"

max_steps = int(1e6)
policy_lr = 0.001
critic_lr = 0.005
model_lr = 0.005
horizon = 1
buffer_size = int(1e5)
gamma = 0.99
save_every = 100
train_horizon = 5
grad_clip = 5.
batch_size = 64

critic_epochs = 1
actor_epochs = 1

seed = 33
h_dim = 64
tau = 0.1
update_target_every = 1

device = torch.device("cpu")
