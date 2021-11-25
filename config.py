import torch
import sys
import getpass

DEBUG = sys.gettrace() is not None
env_id = "inverted_pendulum"

max_steps = int(1e6)
policy_lr = 1e-3
critic_lr = 1e-3
horizon = 200
buffer_size = int(1e5)
gamma = 0.99
save_every = 100
train_horizon = 5
grad_clip = 5.
batch_size = 64
critic_epochs = 1
actor_epochs = 1

regularizer = 1e-4
seed = 33
h_dim = 32
tau = 0.01
update_target_every = 5

device = torch.device("cpu")
if getpass.getuser() != "d3sm0" and torch.cuda.is_available():
    device = torch.device("cuda")
if not DEBUG:
    should_render = False
    proc_num = 5
    host = "mila"
    sweep_yaml = "sweep.yaml"
else:
    should_render = False
    proc_num = 1
    host = ""
    sweep_yaml = ""  # k"sweep.yaml"
