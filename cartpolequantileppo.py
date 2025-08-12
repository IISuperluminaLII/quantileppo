import os
import numpy as np
import matplotlib.pyplot as plt

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import gym

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

# Import your custom QuantilePPO and its policy
from quantileppoimpl.purequantilePPO import QuantilePPO, QuantileActorCriticPolicy


class RewardPlotCallback(BaseCallback):
    """
    Works for single-env or VecEnv.
    Collects episode rewards from Monitor (via 'episode' in info/infos) and can plot them.
    """
    def __init__(self, save_dir: str = "plots", plot_every: int | None = None, rolling: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.plot_every = plot_every
        self.rolling = rolling
        self.ep_rewards: list[float] = []
        self.ep_lengths: list[int] = []
        self.ep_timesteps: list[int] = []
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        # VecEnv case: list of infos
        infos = self.locals.get("infos", None)
        if infos is not None:
            for info in infos:
                if "episode" in info:
                    ep = info["episode"]
                    self.ep_rewards.append(float(ep["r"]))
                    self.ep_lengths.append(int(ep["l"]))
                    self.ep_timesteps.append(self.num_timesteps)
        else:
            # Single-env case: 'info' for the current step
            info = self.locals.get("info", {})
            if isinstance(info, dict) and "episode" in info:
                ep = info["episode"]
                self.ep_rewards.append(float(ep["r"]))
                self.ep_lengths.append(int(ep["l"]))
                self.ep_timesteps.append(self.num_timesteps)

        if self.plot_every and (len(self.ep_rewards) > 0) and (len(self.ep_rewards) % self.plot_every == 0):
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


if __name__ == "__main__":
    env_id = "CartPole-v1"
    seed = 1234

    # --- Training env: single environment, no parallelization ---
    train_env = gym.make(env_id)             # no render during training
    train_env = Monitor(train_env)           # needed so callback sees 'episode' info
    try:
        train_env.reset(seed=seed)
        if hasattr(train_env.action_space, "seed"):
            train_env.action_space.seed(seed)
        if hasattr(train_env.observation_space, "seed"):
            train_env.observation_space.seed(seed)
    except TypeError:
        # old gym API fallback
        if hasattr(train_env, "seed"):
            train_env.seed(seed)

    model = QuantilePPO(
        policy=QuantileActorCriticPolicy,
        env=train_env,                 # SB3 will wrap this in a DummyVecEnv with n_envs=1
        n_quantiles=32,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,                   # with 1 env: 512 samples/update
        batch_size=64,                 # 512/64 = 8 minibatches
        n_epochs=5,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        clip_range=0.2,
        seed=seed,
        gae_lambda=0.95,
    )

    reward_cb = RewardPlotCallback(save_dir="plots", plot_every=None, rolling=10, verbose=1)
    model.learn(total_timesteps=100_000, callback=reward_cb)

    # Save a final plot
    reward_cb.plot(save_path="plots/reward_curve.png")

    # --- Evaluation / visualization on a separate human-render env ---
    eval_env = gym.make(env_id, render_mode="human")
    try:
        eval_env.reset(seed=seed + 1)
    except TypeError:
        pass

    obs, _ = eval_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        # Gym >=0.26 API
        obs, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated
    # Window closes when the process exits; keep it open a moment if needed
