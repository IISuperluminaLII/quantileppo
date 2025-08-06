import torch as th
from typing import Any, Callable, Dict, Optional, Type, Union
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv

from utilities import QuantileHead
from utilities import QuantileDistribution
from utilities import QuantileLoss


class QuantileActorCriticPolicy(ActorCriticPolicy):
    """
    ActorCriticPolicy replacing scalar value head with a quantile head.
    """
    def __init__(
        self,
        *args,
        n_quantiles: int = 32,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_quantiles = n_quantiles
        # Remove default value network
        self.value_net = None
        # Quantile head and loss
        self.quantile_head = QuantileHead(
            input_dim=self.mlp_extractor.latent_dim,
            output_dim=1,
            n_quantiles=self.n_quantiles,
            hidden_dim=self.mlp_extractor.latent_dim,
        )
        self.quantile_loss = QuantileLoss()

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
        # Maintain SB3 signature: returns values, log_prob, entropy, then extras
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

        # Return in standard order for OnPolicyAlgorithm
        return values, log_prob, entropy, quantiles, taus


class QuantilePPO(OnPolicyAlgorithm):
    """
    Quantile-PPO by extending OnPolicyAlgorithm and using quantile loss.
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
        tensorboard_log: Optional[str] = None,
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
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf

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
        clip_range_vf = self.clip_range_vf(self._current_progress_remaining) if self.clip_range_vf else None

        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                # Unpack policy outputs
                values, log_prob, entropy, quantiles, taus = self.policy.evaluate_actions(rollout_data.observations,
                    rollout_data.actions.long() if self.discrete else rollout_data.actions,)

                # Policy loss
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                adv = rollout_data.advantages
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                pg_loss = -th.min(
                    adv * ratio,
                    adv * th.clamp(ratio, 1 - clip_range, 1 + clip_range),
                ).mean()

                # Critic loss via quantile regression
                returns = rollout_data.returns.unsqueeze(-1)
                value_loss = self.policy.quantile_loss(pred=quantiles, target=returns, tau=taus)

                # Entropy
                entropy_loss = -entropy.mean()

                loss = pg_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

