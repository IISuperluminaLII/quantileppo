import gym

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

# Import your custom QuantilePPO and its policy
from quantileppoimpl.piecewisequantilePPO import QuantilePPO, QuantileActorCriticPolicy


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

    # Instantiate QuantilePPO with the custom QuantileActorCriticPolicy
    model = QuantilePPO(
        policy=QuantileActorCriticPolicy,
        env=env,
        n_quantiles=32,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        clip_range=0.2,
        clip_range_vf=None,
        seed=0,
    )

    # Train the model
    model.learn(total_timesteps=25_000)

    # Evaluate the learned policy
    obs = env.reset()
    for _ in range(1_000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
