import itertools

import flax.optim as optim
import haiku as hk
import jax

import config
from envs import jax_envs
from models import Policy, RealDynamics, get_initial_params
from svg import generate_episode, train


def scalars_to_tb(writer, scalars, global_step):
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)


def main():
    key_gen = hk.PRNGSequence(config.seed)
    env = jax_envs.make_env(config.env_id, horizon=config.horizon)
    params = get_initial_params(next(key_gen))

    # dynamics = Dynamics(env, learn_reward=agent_config.learn_reward, std=agent_config.model_std)
    dynamics = RealDynamics(env)

    pi_optim = optim.GradientDescent(learning_rate=config.policy_lr).create(params)

    run(dynamics, env, pi_optim, key_gen)
    config.tb.close()


def run(dynamics, env, pi_optim, key_gen):
    # number of frames
    n_samples = 0
    for global_step in itertools.count():
        params = pi_optim.target
        if n_samples >= config.max_n_samples:
            break
        if global_step % config.save_every == 0:
            print(f"Saved at {global_step}. Progress:{n_samples / config.max_n_samples:.2f}")
            # config.tb.add_object("model", params, global_step)
        trajectory, env_statistics = generate_episode(env, key_gen, params)
        scalars_to_tb(config.tb, env_statistics, n_samples)
        pi_optim, agent_loss = train(dynamics, params, pi_optim, trajectory)
        scalars_to_tb(config.tb, agent_loss, n_samples)
        n_samples += len(trajectory)


if __name__ == "__main__":
    if config.DEBUG:
        with jax.disable_jit():
            main()
    else:
        main()
