from utilities import QuantileDistribution, QuantileHead, QuantileLoss, FocalLoss
from utilities import build_ffnn_model
import torch.nn as nn
import torch


class WorldModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, n_heads, n_layers, n_quantiles, device="cpu", seq_len=5,
                 n_ensemble_heads=5):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.n_quantiles = n_quantiles
        self.seq_len = seq_len
        self.ensemble_variation_coef = 0

        self.mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(device)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            batch_first=True,
            dim_feedforward=hidden_dim * 2,
            norm_first=True
        ).to(device)
        self.input_proj = nn.Linear(state_dim + action_dim, hidden_dim).to(device)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers,
                                                         norm=nn.LayerNorm(hidden_dim)).to(device)

        # Each ensemble head gets its own FFNN embedder
        self.state_ffnns = nn.ModuleList([
            build_ffnn_model(
                n_input_features=hidden_dim,
                n_output_features=hidden_dim,
                use_layer_norm=True,
                layer_widths=[hidden_dim] * n_layers,
                act_fn_callable=nn.PReLU,
                apply_layer_norm_final_layer=True,
                output_act_fn_callable=nn.PReLU,
                device=device
            ) for _ in range(n_ensemble_heads)
        ])
        self.state_pred_head = nn.ModuleList([
            QuantileHead(input_dim=hidden_dim, output_dim=state_dim, hidden_dim=hidden_dim,
                         n_quantiles=n_quantiles, n_basis=64, device=device) for _ in range(n_ensemble_heads)
        ])

        self.reward_ffnns = nn.ModuleList([
            build_ffnn_model(
                n_input_features=hidden_dim,
                n_output_features=hidden_dim,
                use_layer_norm=True,
                layer_widths=[hidden_dim] * n_layers,
                act_fn_callable=nn.PReLU,
                apply_layer_norm_final_layer=True,
                output_act_fn_callable=nn.PReLU,
                device=device
            ) for _ in range(n_ensemble_heads)
        ])
        self.reward_pred_head = nn.ModuleList([
            QuantileHead(input_dim=hidden_dim, output_dim=1, hidden_dim=hidden_dim,
                         n_quantiles=n_quantiles, n_basis=64, device=device) for _ in range(n_ensemble_heads)
        ])

        self.done_ffnns = nn.ModuleList([
            build_ffnn_model(
                n_input_features=hidden_dim,
                n_output_features=hidden_dim,
                use_layer_norm=True,
                layer_widths=[hidden_dim] * n_layers,
                act_fn_callable=nn.PReLU,
                apply_layer_norm_final_layer=True,
                output_act_fn_callable=nn.PReLU,
                device=device
            ) for _ in range(n_ensemble_heads)
        ])
        self.done_pred_head = nn.ModuleList([
            nn.Linear(hidden_dim, 2).to(device) for _ in range(n_ensemble_heads)
        ])

        self.quantile_loss = QuantileLoss()
        self.focal_loss = FocalLoss(gamma=4)
        self.sft = nn.Softmax(dim=-1)

    def embed(self, states, actions, tau=None):
        if tau is None:
            tau = torch.rand(states.size(0), self.n_quantiles, 1, device=self.device)

        input = torch.cat([states, actions], dim=-1)
        x = self.input_proj(input)
        x = self.transformer_encoder(x, mask=self.mask, is_causal=True)
        x = x.mean(dim=1)
        return x, tau

    def get_next_state_preds(self, embedded, tau=None):
        state_error_quantiles = torch.stack([
            head(ffnn(embedded), tau=tau)
            for ffnn, head in zip(self.state_ffnns, self.state_pred_head)
        ], dim=0)
        return state_error_quantiles

    def get_reward_preds(self, embedded, tau=None):
        reward_quantiles = torch.stack([
            head(ffnn(embedded), tau=tau)
            for ffnn, head in zip(self.reward_ffnns, self.reward_pred_head)
        ], dim=0)
        return reward_quantiles

    def get_done_preds(self, embedded):
        done_logits = torch.stack([
            head(ffnn(embedded))
            for ffnn, head in zip(self.done_ffnns, self.done_pred_head)
        ], dim=0)
        return done_logits

    def get_ensemble_outputs(self, embedded, tau=None):
        done_logits = self.get_done_preds(embedded)
        state_error_quantiles = self.get_next_state_preds(embedded, tau=tau)
        reward_quantiles = self.get_reward_preds(embedded, tau=tau)
        return done_logits, state_error_quantiles, reward_quantiles

    def forward(self, states, actions, tau=None):
        x, tau = self.embed(states, actions, tau=tau)
        done_logits, state_error_quantiles, reward_quantiles = self.get_ensemble_outputs(x, tau=tau)
        return self.predict(done_logits, state_error_quantiles, reward_quantiles, tau=tau)

    def predict(self, done_logits, state_error_quantiles, reward_quantiles, tau=None):
        # DONE SHAPE: (ensemble_dim, batch_dim, 2)
        # STATE QUANTILE SHAPE: (ensemble_dim, batch_dim, n_quantiles, state_dim)
        # REWARD QUANTILE SHAPE: (ensemble_dim, batch_dim, n_quantiles, 1)
        # TAU SHAPE: (batch_dim, n_quantiles, 1)
        return done_logits.mean(dim=0), state_error_quantiles.mean(dim=0), reward_quantiles.mean(dim=0), tau

    def compute_disagreement(self, done_logits, state_error_quantiles, reward_quantiles, method="cv"):
        """
        Compute ensemble disagreement using scale-invariant measures.

        Args:
            method: One of ["w1", "relative_w1", "cv", "cosine", "robust"]
        """
        if method == "w1":
            state_divergence = self._average_pairwise_wasserstein1(state_error_quantiles)
            reward_divergence = self._average_pairwise_wasserstein1(reward_quantiles)
        elif method == "relative_w1":
            state_divergence = self._average_pairwise_relative_wasserstein1(state_error_quantiles)
            reward_divergence = self._average_pairwise_relative_wasserstein1(reward_quantiles)
        elif method == "cv":
            state_divergence = self._ensemble_coefficient_of_variation(state_error_quantiles)
            reward_divergence = self._ensemble_coefficient_of_variation(reward_quantiles)
        elif method == "cosine":
            state_divergence = self._average_pairwise_cosine_distance(state_error_quantiles)
            reward_divergence = self._average_pairwise_cosine_distance(reward_quantiles)
        elif method == "robust":
            state_divergence = self._robust_normalized_wasserstein1(state_error_quantiles)
            reward_divergence = self._robust_normalized_wasserstein1(reward_quantiles)
        elif method == "var":
            state_means = state_error_quantiles.mean(dim=-2)
            reward_means = reward_quantiles.mean(dim=-2)
            state_divergence = torch.var(state_means, dim=0)
            reward_divergence = torch.var(reward_means, dim=0)
        else:
            raise ValueError(f"Unknown method: {method}")

        done_divergence = self._average_pairwise_tvd(done_logits)
        return done_divergence, state_divergence, reward_divergence

    def get_loss(self, states, actions, next_state_errors, rewards, dones, tau=None):
        done_logits, state_error_quantiles, reward_quantiles, tau = self.forward(states, actions, tau=tau)

        done_loss = self.focal_loss(done_logits, dones.long().flatten())
        state_loss = self.quantile_loss(pred=state_error_quantiles, target=next_state_errors, tau=tau)
        reward_loss = self.quantile_loss(pred=reward_quantiles, target=rewards, tau=tau)

        state_diversity_loss = self._ensemble_diversity_loss(self.state_pred_head)
        reward_diversity_loss = self._ensemble_diversity_loss(self.reward_pred_head)
        done_diversity_loss = self._ensemble_diversity_loss(self.done_pred_head)
        return done_loss + state_loss + reward_loss + state_diversity_loss + reward_diversity_loss + done_diversity_loss

    def _ensemble_diversity_loss(self, ensemble):
        if self.ensemble_variation_coef == 0:
            return 0
        # Compute average cosine similarity between all pairs of heads (excluding self)
        vecs = [nn.utils.parameters_to_vector(head.parameters()) for head in ensemble]
        vecs = torch.stack(vecs)  # (n_heads, n_params)
        n = vecs.shape[0]
        # Normalize vectors
        vecs_norm = vecs / (vecs.norm(dim=1, keepdim=True) + 1e-8)
        # Compute cosine similarity matrix
        sim_matrix = torch.mm(vecs_norm, vecs_norm.t())  # (n_heads, n_heads)
        # Exclude diagonal (self-similarity)
        mask = ~torch.eye(n, dtype=torch.bool, device=vecs.device)
        avg_sim = sim_matrix[mask].mean()
        return self.ensemble_variation_coef * avg_sim

    def predict_transition(self, state, action, tau=None):
        if tau is None:
            tau = torch.rand(1, self.n_quantiles, 1, device=self.device)

        done_logits, state_error_quantiles, reward_quantiles, tau = self.forward(state, action, tau=tau)
        done_probs = self.sft(done_logits)
        state_error_distribution = QuantileDistribution(state_error_quantiles, tau)
        reward_distribution = QuantileDistribution(reward_quantiles, tau)
        return done_probs.multinomial(1), state + state_error_distribution.sample(), reward_distribution.sample()

    def _average_pairwise_wasserstein1(self, q):
        K, B, N, D = q.shape
        # Expand for broadcasting: (K, K, B, N, D)
        q1 = q[:, None, ...]  # (K, 1, B, N, D)
        q2 = q[None, :, ...]  # (1, K, B, N, D)
        # Pairwise absolute difference: (K, K, B, N, D)
        diff = torch.abs(q1 - q2)
        # Mean over quantiles: (K, K, B, D)
        wasserstein = diff.mean(dim=3)
        # Mean over all pairs (excluding self-comparisons)
        # Option 1: mean over upper triangle only (no repeats, no self)
        mask = ~torch.eye(K, dtype=torch.bool, device=q.device)
        avg = wasserstein[mask].view(K, K - 1, B, D).mean(dim=(0, 1))  # (B, D)
        return avg  # (batch_dim, output_dim)

    def _average_pairwise_relative_wasserstein1(self, q):
        """Scale-invariant W1 distance normalized by prediction magnitude."""
        K, B, N, D = q.shape

        # Compute pairwise W1 distances
        q1 = q[:, None, ...]  # (K, 1, B, N, D)
        q2 = q[None, :, ...]  # (1, K, B, N, D)
        diff = torch.abs(q1 - q2)
        wasserstein = diff.mean(dim=3)  # (K, K, B, D)

        # Compute normalization factor (mean magnitude across all heads)
        mean_magnitude = q.mean(dim=2).mean(dim=0, keepdim=True).mean(dim=1, keepdim=True)  # (1, 1, B, D)

        # Avoid division by zero
        eps = 1e-8
        relative_wasserstein = wasserstein / (torch.abs(mean_magnitude) + eps)

        # Average over pairs
        mask = ~torch.eye(K, dtype=torch.bool, device=q.device)
        avg = relative_wasserstein[mask].view(K, K - 1, B, D).mean(dim=(0, 1))
        return avg

    def _ensemble_coefficient_of_variation(self, q):
        """Coefficient of variation across ensemble heads."""
        K, B, N, D = q.shape

        # Compute mean across quantiles for each head
        head_means = q.mean(dim=2)  # (K, B, D)

        # Compute coefficient of variation across heads
        head_std = torch.std(head_means, dim=0)  # (B, D)
        head_mean = torch.mean(head_means, dim=0)  # (B, D)

        eps = 1e-8
        cv = head_std / (torch.abs(head_mean) + eps)
        return cv

    def _average_pairwise_cosine_distance(self, q):
        """Cosine distance between ensemble predictions (invariant to magnitude)."""
        K, B, N, D = q.shape

        # Flatten quantiles for each head
        q_flat = q.view(K, B, -1)  # (K, B, N*D)

        # Normalize each head's prediction
        q_norm = torch.nn.functional.normalize(q_flat, p=2, dim=2)  # (K, B, N*D)

        # Compute cosine similarity matrix
        cos_sim = torch.bmm(q_norm.permute(1, 0, 2), q_norm.permute(1, 2, 0))  # (B, K, K)

        # Convert to distance (1 - similarity)
        cos_dist = 1 - cos_sim  # (B, K, K)

        # Average over pairs (excluding diagonal)
        mask = ~torch.eye(K, dtype=torch.bool, device=q.device)
        avg_dist = cos_dist[:, mask].view(-1, K, K - 1).mean(dim=2)  # (B,)

        return avg_dist.unsqueeze(-1).expand(-1, D)  # (B, D)

    def _robust_normalized_wasserstein1(self, q):
        """Robust W1 distance using median absolute deviation normalization."""
        K, B, N, D = q.shape

        # Compute median for each dimension
        median = torch.median(q, dim=2)[0]  # (K, B, D)

        # Compute median absolute deviation (MAD)
        mad = torch.median(torch.abs(q - median[:, :, None, :]), dim=2)[0]  # (K, B, D)

        # Robust normalization
        q_normalized = (q - median[:, :, None, :]) / (mad[:, :, None, :] + 1e-8)

        # Compute W1 on normalized quantiles
        return self._average_pairwise_wasserstein1(q_normalized)

    def _average_pairwise_tvd(self, probs):
        # probs: (ensemble_dim, batch_dim, output_dim)
        K, B, D = probs.shape
        # Expand for broadcasting: (K, K, B, D)
        p1 = probs[:, None, ...]  # (K, 1, B, D)
        p2 = probs[None, :, ...]  # (1, K, B, D)
        # Pairwise absolute difference: (K, K, B, D)
        diff = torch.abs(p1 - p2)
        # Sum over output_dim for TVD: (K, K, B)
        tvd = 0.5 * diff.sum(dim=3)
        # Mask out self-comparisons
        mask = ~torch.eye(K, dtype=torch.bool, device=probs.device)
        avg = tvd[mask].view(K, K - 1, B).mean(dim=(0, 1))  # (B,)
        return avg  # (batch_dim,)