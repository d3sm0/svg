import haiku as hk

from sfvg import lib
from config import config
from sfvg.dynamics import FakeModel, Model
from envs import jax_envs
from utils import get_agent


import jax.config
jax.config.update("jax_debug_nans", True)

def main():
    key_gen = hk.PRNGSequence(config.seed)
    env = jax_envs.make_env(key=next(key_gen), env_id=config.env_id, horizon=config.env_horizon)
    if config.learn_model:
        dynamics = Model(r=env.unwrapped.r, env_spec=env.env_spec)
    else:
        dynamics = FakeModel(f=env.unwrapped.f, r=env.unwrapped.r, env_spec=env.env_spec)
    del env
    agent = get_agent()(dynamics)
    lib.train(key_gen, agent, dynamics)


if __name__ == "__main__":
    if config.DEBUG:
        with jax.disable_jit():
            main()
    else:
        main()
