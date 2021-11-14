DEBUG = True

max_steps = int(1e5) / 200
policy_lr = 1e-3
critic_lr = 1e-3
horizon = 200 if not DEBUG else 50
buffer_size = int(1e3)
gamma = 0.99
save_every = 1000
train_horizon = 5
grad_clip = 5.
batch_size = 32
regularizer = 1e-4
seed = 33
h_dim = 32
should_render = False
proc_num = 5
host = "mila"

sweep_yaml ="sweep.yaml"