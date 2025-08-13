import gym
import numpy as np
import os
import matplotlib.pyplot as plt

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

from stable_baselines3.common.utils import set_random_seed
from quantileppoimpl.purequantilePPO import QuantilePPO, QuantileActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback


class SimpleMonitor(gym.Env):
    def __init__(self, env):
        self.env = env
        self.ep_ret = 0.0
        self.ep_len = 0
        self._needs_reset = True

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def reward_range(self):
        return getattr(self.env, "reward_range", (-float("inf"), float("inf")))

    @property
    def spec(self):
        return getattr(self.env, "spec", None)

    def reset(self, **kwargs):
        self.ep_ret = 0.0
        self.ep_len = 0
        self._needs_reset = False
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
            return obs
        return out

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = terminated or truncated
        else:
            obs, reward, done, info = out
        self.ep_ret += float(reward)
        self.ep_len += 1
        if done:
            info = dict(info)
            info["episode"] = {"r": self.ep_ret, "l": self.ep_len}
            self._needs_reset = True
        return obs, reward, done, info

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        if hasattr(self.env, "seed"):
            return self.env.seed(seed)
        # gymnasium-style
        if hasattr(self.env, "reset"):
            return self.env.reset(seed=seed)
        return None


class RewardPlotCallback(BaseCallback):
    def __init__(self, save_dir: str = "plots", plot_every: int | None = None, rolling: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.plot_every = plot_every
        self.rolling = rolling
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_timesteps = []
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos is None:
            info = self.locals.get("info")
            if info is not None:
                infos = [info]
        if infos:
            for info in infos:
                if "episode" in info:
                    ep = info["episode"]
                    self.ep_rewards.append(float(ep["r"]))
                    self.ep_lengths.append(int(ep["l"]))
                    self.ep_timesteps.append(self.num_timesteps)
                    if self.plot_every and (len(self.ep_rewards) % self.plot_every == 0):
                        self.plot(save_path=os.path.join(self.save_dir, f"reward_curve2_{len(self.ep_rewards)}.png"))
        return True

    def plot(self, save_path: str | None = None, show: bool = False):
        if not self.ep_rewards:
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

    def make_env():
        env = gym.make(env_id)
        try:
            env.reset(seed=seed)
            if hasattr(env.action_space, "seed"):
                env.action_space.seed(seed)
            if hasattr(env.observation_space, "seed"):
                env.observation_space.seed(seed)
        except TypeError:
            if hasattr(env, "seed"):
                env.seed(seed)
            if hasattr(env.action_space, "seed"):
                env.action_space.seed(seed)
        return env

    def get_incremented_path(path):
        base, ext = os.path.splitext(path)
        counter = 1
        new_path = path
        while os.path.exists(new_path):
            new_path = f"{base}_{counter}{ext}"
            counter += 1
        return new_path

    train_env = SimpleMonitor(make_env())

    model = QuantilePPO(
        policy=QuantileActorCriticPolicy,
        env=train_env,
        n_quantiles=32,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=5,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        clip_range=0.2,
        seed=seed,
        gae_lambda=0.95,
        device="cuda",
    )

    reward_cb = RewardPlotCallback(save_dir="../plots", plot_every=None, rolling=10, verbose=1)
    model.learn(total_timesteps=5_000_000, callback=reward_cb)

    # Usage
    save_path = get_incremented_path("../plots/reward_curve_randomtau2.png")
    reward_cb.plot(save_path=save_path)

    obs = train_env.reset()
    for _ in range(1_000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = train_env.step(action)
        train_env.render()
        if done:
            obs = train_env.reset()
