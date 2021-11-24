import torch

DEBUG = True  # sys.gettrace() is not None

max_steps = int(1e5)
policy_lr = 1e-3
critic_lr = 1e-3
horizon = 1000
buffer_size = int(1e4)
gamma = 0.99
save_every = 100
train_horizon = 5
grad_clip = 5.
batch_size = 64
critic_epochs = 5
actor_epochs = 5

regularizer = 1e-4
seed = 33
h_dim = 32
tau = 0.5
update_target_every = 10

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if not DEBUG:
    should_render = False
    proc_num = 5
    host = "mila"
    sweep_yaml = "sweep.yaml"
else:
    should_render = True
    proc_num = 1
    host = ""
    sweep_yaml = ""  # k"sweep.yaml"
