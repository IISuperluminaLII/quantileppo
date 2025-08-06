# new_canvas_quantile_ppo_demo.py

import gym
import numpy as np

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

# Import your QuantilePPO (which extends OnPolicyAlgorithm)
from quantilePPO import QuantilePPO

def make_env(env_id: str, rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param rank: (int) index of the subprocess
    :param seed: (int) the initial seed for RNG
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':
    env_id = "CartPole-v1"
    num_cpu = 4  # Number of processes to use

    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    # Instantiate QuantilePPO with 32 quantiles (you can tweak n_quantiles)
    model = QuantilePPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        n_quantiles=32,
        # you can also pass any standard policy_kwargs here, e.g.:
        # policy_kwargs=dict(net_arch=[dict(pi=[64,64], vf=[64,64])]),
        seed=0,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        clip_range=0.2,
        clip_range_vf=None,
    )

    # Train
    model.learn(total_timesteps=25_000)

    # Evaluate
    obs = env.reset()
    for _ in range(1_000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
