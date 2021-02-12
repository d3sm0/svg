import datetime
import os
import sys


class config:
    env_id = "lqg"
    project = "svg"
    seed = 2323
    gamma = .99
    l1_penalty = 1e-4
    lr = 3e-3
    model_lr = 1e-3
    buffer_size = int(1e2)
    batch_size = 32
    model_std = 0.
    learned_model_std = 1.0
    agent_std = 0.1
    DEBUG = True if sys.gettrace() is not None else False
    dtm = datetime.datetime.now().strftime("%d-%H-%M-%S-%f")
    learn_model = False
    eval_every = int(1e3)
    n_training_steps = 3
    model_training_steps = 10
    max_samples = int(1e6)
    grad_clip = 1.0
    hidden_size = 32

    if DEBUG:
        env_horizon = 30
        sample_horizon = 30
        log_dir = os.path.join("debug", dtm)
        ckpt_dir = os.path.join("debug", dtm)
    else:
        env_horizon = 200
        sample_horizon = 200
        log_dir = os.path.join("logs", env_id, dtm)
        ckpt_dir = os.path.join("logs", env_id, dtm)


if config.DEBUG:
    print('''
   			   /~\   Debug mode again, eh?!
                          (O O) _/                 
                          _\=/_                         
          ___            /  _  \                         
         / ()\          //|/.\|\\                        
       _|_____|_       ||  \_/  ||                       
      | | === | |      || |\ /| ||                       
      |_|  O  |_|       # \_ _/ #                        
       ||  O  ||          | | |                          
       ||__*__||          | | |                          
      |~ \___/ ~|         []|[]                          
      /=\ /=\ /=\         | | |                          
______[_]_[_]_[_]________/_]_[_\_____ 
    ''')
