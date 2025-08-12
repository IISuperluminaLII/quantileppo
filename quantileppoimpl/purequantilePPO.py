import torch as th
from torch import nn
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Type, Union
from gym import spaces
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.utils import FloatSchedule

from utilities import QuantileHead
from utilities import QuantileDistribution
from utilities import QuantileLoss


class QuantileActorCriticPolicy(ActorCriticPolicy):
    """
    ActorCriticPolicy replacing scalar value head with a quantile head.
    """
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        n_quantiles: int = 32,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = -2.0,
        full_std: bool = True,
        sde_net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        _init_setup_model: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            squash_output=squash_output,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

        self.n_quantiles = n_quantiles
        # Remove scalar value network
        self.value_net = None
        # Quantile head and quantile loss
        self.quantile_head = QuantileHead(
            input_dim=self.mlp_extractor.latent_dim_vf,
            output_dim=1,
            n_quantiles=self.n_quantiles,
            n_basis=64,
            hidden_dim=self.mlp_extractor.latent_dim_vf,
            device="cuda",
        )
        self.quantile_loss = QuantileLoss()

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs)
        _, latent_vf = self.mlp_extractor(features)
        B = latent_vf.shape[0]
        taus = th.rand(B, self.quantile_head.n_quantiles, 1, device=latent_vf.device)
        q = self.quantile_head(latent_vf, tau=taus)  # [B, N, 1]
        v = q.mean(dim=-2).squeeze(-1)  # [B]
        return v.unsqueeze(-1)  # [B, 1]

    def forward(self, obs: th.Tensor, deterministic: bool = False):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        dist = self._get_action_dist_from_latent(latent_pi)
        actions = dist.get_actions(deterministic=deterministic)
        log_prob = dist.log_prob(actions)

        batch_size = obs.size(0)
        taus = th.rand(batch_size, self.n_quantiles, 1, device=obs.device)
        quantiles = self.quantile_head(latent_vf, tau=taus)

        q_dist = QuantileDistribution(quantile_values=quantiles, tau=taus)
        values = q_dist.mean

        return actions, values, log_prob

    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
    ):
        # Returns (values, log_prob, entropy, quantiles, taus, q_dist)
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        dist = self._get_action_dist_from_latent(latent_pi)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()

        batch_size = obs.size(0)
        taus = th.rand(batch_size, self.n_quantiles, 1, device=obs.device)
        quantiles = self.quantile_head(latent_vf, tau=taus)

        q_dist = QuantileDistribution(quantile_values=quantiles, tau=taus)
        values = q_dist.mean

        return values, log_prob, entropy, quantiles, taus, q_dist


class QuantilePPO(OnPolicyAlgorithm):
    """
    Quantile-PPO by extending OnPolicyAlgorithm and using quantile regression and distribution log-prob.
    """
    def __init__(
        self,
        policy: Union[str, Type[QuantileActorCriticPolicy]] = QuantileActorCriticPolicy,
        env: Union[GymEnv, str] = None,
        n_quantiles: int = 32,
        learning_rate: Union[float, Callable[[float], float]] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Callable[[float], float]] = 0.2,
        clip_range_vf: Optional[Union[float, Callable[[float], float]]] = None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = "tensorboards/",
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        default_policy_kwargs = dict(n_quantiles=n_quantiles)
        if policy_kwargs:
            default_policy_kwargs.update(policy_kwargs)

        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=False,
            sde_sample_freq=-1,
            tensorboard_log=tensorboard_log,
            policy_kwargs=default_policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,
        )

        self.n_quantiles = n_quantiles
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = FloatSchedule(clip_range) #callable float smh
        self.discrete = isinstance(self.action_space, spaces.Discrete)

        self.rollout_buffer = RolloutBuffer(
            buffer_size=self.n_steps,
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        if _init_setup_model:
            self._setup_model()

    def train(self) -> None:
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        # clip_range_vf = self.clip_range_vf(self._current_progress_remaining) if self.clip_range_vf else None

        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                # Evaluate actions and obtain q_dist
                values, log_prob, entropy, quantiles, taus, q_dist = self.policy.evaluate_actions(
                    rollout_data.observations, rollout_data.actions.long() if self.discrete else rollout_data.actions,)

                # Policy gradient loss
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                adv = rollout_data.advantages
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                pg_loss = -th.min(
                    adv * ratio,
                    adv * th.clamp(ratio, 1 - clip_range, 1 + clip_range),
                ).mean()

                # Critic: quantile regression loss only
                returns = rollout_data.returns.unsqueeze(-1)
                value_loss = self.policy.quantile_loss(pred=quantiles, target=returns, tau=taus)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                loss = pg_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                if self.logger is not None:
                    self.logger.record("train/quantile_loss", value_loss.item())

                    # Log quantile distribution statistics
                    quantile_mean = quantiles.mean().item()
                    quantile_std = quantiles.std().item()
                    quantile_min = quantiles.min().item()
                    quantile_max = quantiles.max().item()

                    self.logger.record("train/quantile_mean", quantile_mean)
                    self.logger.record("train/quantile_std", quantile_std)
                    self.logger.record("train/quantile_min", quantile_min)
                    self.logger.record("train/quantile_max", quantile_max)

                    if len(self.ep_info_buffer) > 0:
                        ep_rew_mean = float(np.mean([ep["r"] for ep in self.ep_info_buffer]))
                        ep_len_mean = float(np.mean([ep["l"] for ep in self.ep_info_buffer]))
                        self.logger.record("rollout/ep_rew_mean", ep_rew_mean)
                        self.logger.record("rollout/ep_len_mean", ep_len_mean)

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
