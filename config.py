import sys
#from datetime import datetime

import mila_tools
import torch
#import torch.utils.tensorboard as tb

RUN_SWEEP = 0
REMOTE = 0
NUM_PROCS = 1

sweep_yaml = "sweep.yaml" if RUN_SWEEP else False
HOST = "mila" if REMOTE else ""  # in host
DEBUG = '_pydev_bundle.pydev_log' in sys.modules.keys()
seed = 0
use_cuda = False

mila_tools.register(locals())
device = torch.device("cuda" if use_cuda else "cpu")

################################################################
# Derivative parameters
################################################################
# esh = """
# #SBATCH --mem=24GB
# """
esh = """
#SBATCH --job-name=spython
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH --time=2-00:00
#SBATCH --mem=24GB
#SBATCH --cpus-per-task=4
#SBATCH --partition=long
#SBATCH --get-user-env=L
"""
tb = mila_tools.deploy(host=HOST, sweep_yaml=sweep_yaml, extra_slurm_headers=esh, proc_num=NUM_PROCS)
#dtm = datetime.now().strftime("%d-%H-%M-%S-%f")
#tb = tb.SummaryWriter(log_dir=f"logs/{dtm}")
#