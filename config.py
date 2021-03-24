import sys

import experiment_buddy

RUN_SWEEP = 0
REMOTE = 0
NUM_PROCS = 1

sweep_yaml = "sweep.yaml" if RUN_SWEEP else False
HOST = "mila" if REMOTE else ""  # in host
DEBUG = '_pydev_bundle.pydev_log' in sys.modules.keys()
seed = 0
use_cuda = False

h_dim = 32
max_n_samples = int(1e6)
policy_lr = 1e-3
model_lr = 1e-3
model_std = 0.1
horizon = 200
env_id = "DifferentiablePendulum-v0"
gamma = 0.99
save_every = 100

experiment_buddy.register(locals())
# device = torch.device("cuda" if use_cuda else "cpu")

################################################################
# Derivative parameters
################################################################
# esh = """
# #SBATCH --mem=24GB
# """
tb = experiment_buddy.deploy(host=HOST, sweep_yaml=sweep_yaml, proc_num=NUM_PROCS)
