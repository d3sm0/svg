import sys

import torch

DEBUG = sys.gettrace() is not None
env_id = "lqr"
should_render = False
deploy = False
if deploy == True and not DEBUG:
    proc_num = 5
    host = "mila"
    sweep_yaml = "sweep.yaml"
else:
    proc_num = 1
    host = ""
    sweep_yaml = ""

max_steps = int(5e3)
policy_lr = 0.008
critic_lr = 0.005
model_lr = 0.005
buffer_size = int(1e5)
gamma = 0.99
plan_horizon = 5
train_horizon = 5
grad_clip = 5.
batch_size = 32

critic_epochs = 1
actor_epochs = 1

seed = 33
h_dim = 64
# DDPG
tau = 0.1
update_target_every = 1

device = torch.device("cpu")
