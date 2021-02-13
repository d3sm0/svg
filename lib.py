import itertools

#import flax
#import jax.numpy as jnp
#import optax
#import tensorboardX as tensorboard
#from flax.training import checkpoints
#
#from config import config
#from sfvg.dynamics import Buffer
#from envs import jax_envs
#from utils import Transition, Trajectory, eval_policy
#

def collect(env, agent, params):
    trajectory = Trajectory()
    state, _, done, info = env.get_state()
    t = 0
    while not done and t <= config.sample_horizon:
        action, extra = agent.act(params, state)
        next_state, reward, done, info = env.step(action)
        transition = Transition(state, action, reward, next_state, model_noise=info["model_noise"],
                                action_noise=extra["action_noise"])
        trajectory.append(transition)
        state = next_state
        t += 1
    return trajectory, {"collect/returns": info["returns"]}




def train(key_gen, agent, dynamics):
    writer = tensorboard.SummaryWriter(log_dir=config.log_dir)
    env = jax_envs.make_env(key=next(key_gen), env_id=config.env_id, horizon=config.env_horizon)

    actor_params = agent.initial_params(next(key_gen))
    model_params = dynamics.initial_params(next(key_gen))
    # actor_params = flax.core.FrozenDict(checkpoints.restore_checkpoint("logs/22-21-30-59-884762",target=None, prefix="actor_"))
    # model_params = flax.core.FrozenDict(checkpoints.restore_checkpoint("logs/22-21-30-59-884762",target=None, prefix="model_"))
    # model_params = checkpoints.restore_checkpoint("logs/22-21-14-46-716467", 10)

    buffer = Buffer()
    metrics = {}

    optimizer_init, optimizer = optax.adam(learning_rate=config.lr)
    optimizer_state = optimizer_init(actor_params)

    # optimizer = flax.optim.Adam(learning_rate=config.lr).create(actor_params)
    model_optimizer = flax.optim.Adam(learning_rate=config.model_lr).create(model_params)

    for global_step in itertools.count():
        if global_step % config.eval_every == 0 and not config.DEBUG:
            eval_metrics = eval_policy(env, agent.control_fn, actor_params)
            metrics["eval/returns"] = eval_metrics["returns"]
            checkpoints.save_checkpoint(config.ckpt_dir, actor_params, global_step, prefix="actor_")
            checkpoints.save_checkpoint(config.ckpt_dir, model_params, global_step, prefix="model_")

        trajectory, sample_metrics = collect(env, agent, actor_params)
        metrics.update(**sample_metrics)
        buffer.append(trajectory)
        if config.learn_model:
            for _ in range(config.model_training_steps):
                batch = buffer.sample(next(key_gen), batch_size=config.batch_size)
                model_loss, grads = dynamics.training_step(model_params, batch)
                model_optimizer = model_optimizer.apply_gradient(grads)
                model_params = model_optimizer.target
                metrics["model/loss"] = model_loss

        batch, _ = trajectory.get_full_trajectory(gamma=config.gamma)
        if global_step > config.min_training:  # or config.DEBUG or config.learn_model is False:
            loss_ = 0
            grad_norm = 0
            for _ in range(config.n_training_steps):
                grads, agent_metrics = agent.training_step(actor_params, model_params, batch, sample_metrics["collect/returns"])
                if jnp.isinf(agent_metrics["grad_norm"]) or jnp.isnan(agent_metrics["grad_norm"]):
                    # import ipdb; ipdb.set_trace()
                    grads, agent_metrics = agent.training_step(actor_params, model_params, batch, sample_metrics["collect/returns"])
                grads, optimizer_state = optimizer(grads, optimizer_state)
                actor_params = optax.apply_updates(actor_params, grads)
                loss_ += agent_metrics["loss"]
                grad_norm += agent_metrics["grad_norm"]
            metrics["train/loss"] = loss_ / config.n_training_steps
            metrics["train/grad_norm"] = grad_norm / config.n_training_steps
            agent.zero_grad()

        msg = ""
        for k, v in metrics.items():
            writer.add_scalar(f"train/{k}", v, global_step)
            msg += f"{k}:{float(v):.4f}\t"
        print(msg)
