import torch

DEBUG = True  # sys.gettrace() is not None
env_id = "inverted_double_pendulum"

max_steps = int(1e6)
policy_lr = 1e-4
critic_lr = 1e-3
horizon = 1000
buffer_size = int(1e5)
gamma = 0.99
save_every = 100
train_horizon = 5
grad_clip = 5.
batch_size = 256
critic_epochs = 1
actor_epochs = 1

regularizer = 1e-4
seed = 33
h_dim = 256
tau = 0.001
update_target_every = 1

device = torch.device("cpu")
# if getpass.getuser() != "d3sm0" and torch.cuda.is_available():
#     device = torch.device("cuda")
if not DEBUG:
    should_render = False
    host = "mila"
    sweep_yaml = "sweep.yaml"
    proc_num = 5 if len(sweep_yaml) else 1
else:
    should_render = False
    proc_num = 1
    host = ""
    sweep_yaml = ""  # k"sweep.yaml"
