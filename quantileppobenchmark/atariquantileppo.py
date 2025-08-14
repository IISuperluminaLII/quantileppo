import os
import numpy as np
import gym

import gymnasium as gym
import ale_py  # ensures firing up ROM registration
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import BaseCallback

# Your impl
from quantileppoimpl.purequantilePPO import QuantilePPO, QuantileActorCriticPolicy, QuantileActorCriticCnnPolicy

# matplotlib only needed for the callback's plot method
import matplotlib.pyplot as plt
import gymnasium  # do not alias as 'gym' to avoid conflicts
import ale_py  # required so ALE Atari envs register with Gymnasium

from gymnasium.wrappers import AtariPreprocessing, TransformObservation
from gymnasium.wrappers import AtariPreprocessing, RecordVideo
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor, SubprocVecEnv

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_


class SimpleMonitor(gym.Env):
    def __init__(self, env):
        self.env = env
        self.ep_ret = 0.0
        self.ep_len = 0

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self, **kwargs):
        self.ep_ret = 0.0
        self.ep_len = 0
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
        return obs, reward, done, info

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()

class PongScoreWrapper(gym.Wrapper):
    """
    Tracks the true Pong scoreboard and writes into info:
      info["score_agent"], info["score_opponent"]
    Compatible with Gym (4-tuple) and Gymnasium (5-tuple).
    """
    def __init__(self, env):
        super().__init__(env)
        self.agent_score = 0
        self.opp_score = 0

    def reset(self, **kwargs):
        self.agent_score = 0
        self.opp_score = 0
        out = self.env.reset(**kwargs)
        # If Gymnasium reset returns (obs, info), pass through and seed info with scores
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
            info = dict(info)
            info["score_agent"] = 0
            info["score_opponent"] = 0
            return obs, info
        return out  # classic Gym returns just obs

    def step(self, action):
        out = self.env.step(action)

        def _update(reward, info):
            if reward != 0:
                pts = int(round(abs(float(reward))))
                if reward > 0:
                    self.agent_score += pts
                else:
                    self.opp_score += pts
            info = dict(info)
            info["score_agent"] = self.agent_score
            info["score_opponent"] = self.opp_score
            return info

        # Gymnasium: 5-tuple
        if isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
            info = _update(reward, info)
            return obs, reward, terminated, truncated, info

        # Classic Gym: 4-tuple
        obs, reward, done, info = out
        info = _update(reward, info)
        return obs, reward, done, info


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

def _to_ale_id(env_id: str) -> str:
    # minimal mapping for common old Gym IDs
    mapping = {
        "PongNoFrameskip-v4": "ALE/Pong-v5",
        "BreakoutNoFrameskip-v4": "ALE/Breakout-v5",
        "SpaceInvadersNoFrameskip-v4": "ALE/SpaceInvaders-v5",
    }
    return mapping.get(env_id, env_id)


def make_atari_env(env_id="ALE/Pong-v5", seed=1234, n_stack=4, scale_obs=True):
    env_id = _to_ale_id(env_id)
    """
    Returns a DummyVecEnv whose observation space is (H, W, C) with C = n_stack (grayscale)
    or 3*n_stack (color). No VecTransposeImage is used.
    """
    def _make():
        env = gym.make(env_id, frameskip=1, render_mode="human")  # disable frame-skip here

        # Record every episode — you can change lambda e: e % 10 == 0 to record only some
        # env = RecordVideo(env, video_folder="videos/", episode_trigger=lambda e: e % 1000 == 0)

        env = AtariPreprocessing(env,noop_max=30, frame_skip=4, screen_size=84,
                                 grayscale_obs=True, terminal_on_life_loss=True, scale_obs=False)

        # Insert scoreboard writer so info has keys for VecMonitor. SB3 is stable, but man is it shyte
        env = PongScoreWrapper(env)
        return env

    vec = DummyVecEnv([_make])
    vec = VecFrameStack(vec, n_stack=n_stack)  # stacks along channel dimension
    vec = VecMonitor(vec, info_keywords=("score_agent", "score_opponent"))
    return vec


if __name__ == "__main__":
    # === Choose your Atari game ===
    # Common choices: "PongNoFrameskip-v4", "BreakoutNoFrameskip-v4", "SpaceInvadersNoFrameskip-v4"
    env_id = "PongNoFrameskip-v4"
    seed = 1234

    train_env = make_atari_env(env_id=env_id, seed=seed)

    # ===== Atari-tuned hyperparameters for QuantilePPO on pixels =====
    # Notes:
    # - Learning rate & clip adjusted for stability on high-dim obs
    # - n_steps small (128–256) for on-policy variance control without vec env
    # - batch_size large enough to use multiple epochs (3–4) without overfitting
    # - ent_coef >0 encourages exploration on sparse-reward games
    # - more quantiles for pixel tasks (32 or 64). Start with 32.
    model = QuantilePPO(
        policy=QuantileActorCriticCnnPolicy,
        env=train_env,
        n_quantiles=32,
        verbose=1,

        # Optim & rollout
        learning_rate=2.5e-4,      # linear-ish good default for Atari
        n_steps=256,               # 128–256 works; increase if you vectorize envs
        batch_size=256,            # must be divisible by n_steps * (#envs); here single env so any power of 2 is fine
        n_epochs=3,                # 3–4 for PPO on Atari
        max_grad_norm=0.5,

        # PPO loss mix
        clip_range=0.1,            # tighter clip generally steadier on pixels
        gae_lambda=0.95,           # standard Atari setting
        vf_coef=0.5,
        ent_coef=0.01,             # encourage exploration; try 0.005–0.02 per game

        seed=seed,
        device="cuda",
    )

    reward_cb = RewardPlotCallback(save_dir="../plots", plot_every=None, rolling=20, verbose=1)

    # Atari is slow; real runs are 5–10M+ timesteps. Start smaller to verify learning curve.
    model.learn(total_timesteps=2_000_000, callback=reward_cb)
    reward_cb.plot(save_path=f"plots/reward_curve_{env_id}.png")

    # Quick greedy run
    obs = train_env.reset()
    for _ in range(10_000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = train_env.step(action)
        # train_env.render()  # enable if you really want to watch; slows a lot
        if done:
            obs = train_env.reset()
