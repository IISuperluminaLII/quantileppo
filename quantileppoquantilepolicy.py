import os
import numpy as np
import gym
import matplotlib.pyplot as plt


from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

# Import your custom QuantilePPO and its policy
from quantileppoimpl.purequantilePPO import QuantilePPO, QuantileActorCriticPolicy

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback


def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        # New API (Gym >= 0.26 / Gymnasium)
        try:
            env.reset(seed=seed + rank)
            if hasattr(env.action_space, "seed"):
                env.action_space.seed(seed + rank)
            if hasattr(env.observation_space, "seed"):
                env.observation_space.seed(seed + rank)
        except TypeError:
            # Old Gym API
            if hasattr(env, "seed"):
                env.seed(seed + rank)
            if hasattr(env.action_space, "seed"):
                env.action_space.seed(seed + rank)
        return env
    return _init

#add simple reward callback based on existing sb3 reward callbacks
class RewardPlotCallback(BaseCallback):
    """
    Collects episode rewards from Monitor-wrapped envs and can plot them.
    Works with VecEnvs (reads 'episode' from infos when an env finishes).
    """
    def __init__(self, save_dir: str = "plots", plot_every: int | None = None, rolling: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.plot_every = plot_every  # set to an int to auto-save every N episodes
        self.rolling = rolling
        self.ep_rewards: list[float] = []
        self.ep_lengths: list[int] = []
        self.ep_timesteps: list[int] = []
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        # 'infos' is a list (one per env); Monitor puts 'episode' when that env just ended
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep = info["episode"]
                self.ep_rewards.append(float(ep["r"]))
                self.ep_lengths.append(int(ep["l"]))
                self.ep_timesteps.append(self.num_timesteps)

                if self.plot_every and (len(self.ep_rewards) % self.plot_every == 0):
                    self.plot(save_path=os.path.join(self.save_dir, f"reward_curve_{len(self.ep_rewards)}.png"))
        return True

    def plot(self, save_path: str | None = None, show: bool = False):
        if not self.ep_rewards:
            if self.verbose:
                print("No episodes recorded yet.")
            return
        rews = np.asarray(self.ep_rewards, dtype=float)
        xs = np.arange(1, len(rews) + 1)

        plt.figure()
        plt.plot(xs, rews, label="Episode return")
        if len(rews) >= self.rolling:
            ma = np.convolve(rews, np.ones(self.rolling) / self.rolling, mode="valid")
            plt.plot(np.arange(self.rolling, len(rews) + 1), ma, label=f"Rolling mean ({self.rolling})")
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title("Training Reward")
        plt.grid(True, alpha=0.3)
        plt.legend()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()



# --- in your main block, wire the callback and plot after training ---
if __name__ == '__main__':
    env_id = "CartPole-v1"
    num_cpu = 4
    env = SubprocVecEnv([make_env(env_id, i, 1234) for i in range(num_cpu)])

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
        seed=1234,
    )

    reward_cb = RewardPlotCallback(save_dir="plots", plot_every=None, rolling=10, verbose=1)
    model.learn(total_timesteps=25_000, callback=reward_cb)

    # Save a final plot
    reward_cb.plot(save_path="plots/reward_curve.png")

    # (optional) quick eval loop (render may be a no-op on VecEnvs)
    obs = env.reset()
    for _ in range(1_000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

