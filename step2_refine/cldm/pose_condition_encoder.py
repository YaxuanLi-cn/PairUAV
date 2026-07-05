"""
DreamNav pose-condition encoders for SD/ControlNet cross-attention.

This is adapted for the current DreamNav Step2 code path:
    heading_num, range_num -> pose tokens -> c_crossattn

It intentionally does NOT use the UniLIP/DiT-style inject_pose_token(hidden_states,
attention_mask, ...). In this repo, ControlLDMNumeric already feeds c_crossattn
to ControlNet and UNet SpatialTransformer blocks, so the encoder keeps the same
call signature as NumericConditionEncoderSimple:
    forward(heading_num, range_num) -> (B, seq_len, context_dim)

Modes:
    continuous: sin/cos + Fourier/timestep-style features + MLP
    discrete:   heading/range bins -> embedding tables -> MLP
    hybrid:     discrete embedding + continuous residual -> MLP
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class DreamNavPoseConditionEncoder(nn.Module):
    """Encode heading/range into cross-attention pose tokens.

    Parameters
    ----------
    context_dim:
        Cross-attention context dimension used by SD1.5, normally 768.
    hidden_dim:
        Internal MLP width.
    seq_len:
        Number of pose tokens returned to cross-attention. For pose-only
        conditioning, 1--8 tokens is usually cleaner than text's 77 tokens.
    mode:
        'continuous', 'discrete', or 'hybrid'.
    num_heading_bins:
        Number of circular heading bins. 72 means 5 degrees/bin, which can
        distinguish pred-10, pred, pred+10 candidates.
    num_range_bins:
        Number of range bins over [range_min, range_max]. Continuous residual
        remains active in hybrid mode, so range precision is not limited by bins.
    range_min, range_max:
        Range clipping interval. Adjust these after checking train-set statistics.
    num_frequencies:
        Fourier frequencies for continuous features. 4 gives frequencies
        1, 2, 4, 8.
    pose_alpha_init:
        Initial strength of the pose token. Kept as a trainable scalar by default.
    learnable_alpha:
        Whether pose_alpha is trainable.
    use_layernorm:
        Use LayerNorm instead of tanh saturation.
    """

    def __init__(
        self,
        context_dim: int = 768,
        hidden_dim: int = 256,
        seq_len: int = 4,
        mode: str = "hybrid",
        num_heading_bins: int = 72,
        num_range_bins: int = 128,
        range_min: float = 0.0,
        range_max: float = 256.0,
        num_frequencies: int = 4,
        pose_alpha_init: float = 1.0,
        learnable_alpha: bool = True,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()
        mode = mode.lower().strip()
        if mode not in {"continuous", "discrete", "hybrid"}:
            raise ValueError(f"Unsupported pose encoder mode: {mode}")
        if range_max <= range_min:
            raise ValueError("range_max must be larger than range_min")

        self.context_dim = int(context_dim)
        self.hidden_dim = int(hidden_dim)
        self.seq_len = int(seq_len)
        self.mode = mode
        self.num_heading_bins = int(num_heading_bins)
        self.num_range_bins = int(num_range_bins)
        self.range_min = float(range_min)
        self.range_max = float(range_max)
        self.num_frequencies = int(num_frequencies)

        self.use_discrete = mode in {"discrete", "hybrid"}
        self.use_continuous = mode in {"continuous", "hybrid"}

        if self.use_discrete:
            self.heading_emb = nn.Embedding(self.num_heading_bins, hidden_dim)
            self.range_emb = nn.Embedding(self.num_range_bins, hidden_dim)
            self.discrete_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.SiLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )

        if self.use_continuous:
            # heading Fourier: sin/cos(k theta), range Fourier: sin/cos(k*pi*r_norm), plus raw range_norm.
            cont_dim = 4 * self.num_frequencies + 1
            self.continuous_mlp = nn.Sequential(
                nn.Linear(cont_dim, hidden_dim * 2),
                nn.SiLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )

        if mode == "hybrid":
            fuse_in = hidden_dim * 2
        else:
            fuse_in = hidden_dim
        self.fuse_mlp = nn.Sequential(
            nn.Linear(fuse_in, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, context_dim),
        )

        # Expand one fused pose vector into a short pose-token sequence.
        self.to_tokens = nn.Linear(context_dim, context_dim * self.seq_len)
        self.pos_embed = nn.Parameter(torch.randn(1, self.seq_len, context_dim) * 0.02)
        self.norm = nn.LayerNorm(context_dim) if use_layernorm else nn.Identity()

        alpha = torch.tensor(float(pose_alpha_init), dtype=torch.float32)
        if learnable_alpha:
            self.pose_alpha = nn.Parameter(alpha)
        else:
            self.register_buffer("pose_alpha", alpha)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _as_vector(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = x.float()
        if x.dim() == 0:
            x = x.view(1)
        if x.dim() > 1:
            x = x.view(x.shape[0], -1)[:, 0]
        return x

    def heading_to_bin(self, heading_deg: torch.Tensor) -> torch.LongTensor:
        # Support either [-180, 180] or arbitrary degrees by circular modulo.
        h = torch.remainder(heading_deg, 360.0)
        bin_size = 360.0 / float(self.num_heading_bins)
        # round-to-nearest bin center rather than always floor.
        idx = torch.floor((h + bin_size / 2.0) / bin_size).long()
        return torch.remainder(idx, self.num_heading_bins)

    def range_to_bin(self, range_val: torch.Tensor) -> torch.LongTensor:
        r = range_val.clamp(self.range_min, self.range_max)
        frac = (r - self.range_min) / (self.range_max - self.range_min)
        idx = torch.floor(frac * self.num_range_bins).long()
        return idx.clamp(0, self.num_range_bins - 1)

    def _continuous_features(self, heading_deg: torch.Tensor, range_val: torch.Tensor) -> torch.Tensor:
        dtype = heading_deg.dtype
        device = heading_deg.device
        theta = heading_deg / 180.0 * math.pi
        r = range_val.clamp(self.range_min, self.range_max)
        # Normalize range to [-1, 1].
        r_norm = 2.0 * (r - self.range_min) / (self.range_max - self.range_min) - 1.0

        freqs = torch.pow(
            torch.tensor(2.0, device=device, dtype=dtype),
            torch.arange(self.num_frequencies, device=device, dtype=dtype),
        )
        # (B, F)
        theta_f = theta[:, None] * freqs[None, :]
        range_f = math.pi * r_norm[:, None] * freqs[None, :]
        feats = [
            torch.sin(theta_f), torch.cos(theta_f),
            torch.sin(range_f), torch.cos(range_f),
            r_norm[:, None],
        ]
        return torch.cat(feats, dim=-1)

    def forward(self, heading_num: torch.Tensor, range_num: torch.Tensor) -> torch.Tensor:
        heading_num = self._as_vector(heading_num).to(next(self.parameters()).device)
        range_num = self._as_vector(range_num).to(device=heading_num.device, dtype=heading_num.dtype)
        batch_size = heading_num.shape[0]

        parts = []
        if self.use_discrete:
            h_idx = self.heading_to_bin(heading_num)
            r_idx = self.range_to_bin(range_num)
            h_emb = self.heading_emb(h_idx)
            r_emb = self.range_emb(r_idx)
            disc = self.discrete_mlp(torch.cat([h_emb, r_emb], dim=-1))
            parts.append(disc)

        if self.use_continuous:
            cont_feat = self._continuous_features(heading_num, range_num)
            cont = self.continuous_mlp(cont_feat.to(dtype=next(self.parameters()).dtype))
            parts.append(cont)

        if len(parts) == 1:
            fused_in = parts[0]
        else:
            fused_in = torch.cat(parts, dim=-1)
        fused = self.fuse_mlp(fused_in)
        tokens = self.to_tokens(fused).view(batch_size, self.seq_len, self.context_dim)
        tokens = self.norm(tokens + self.pos_embed.to(tokens.dtype))
        alpha = self.pose_alpha.to(device=tokens.device, dtype=tokens.dtype).clamp(0.0, 10.0)
        tokens = alpha * tokens
        tokens = torch.nan_to_num(tokens, nan=0.0, posinf=10.0, neginf=-10.0)
        return tokens

    def get_unconditional_conditioning(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Return zero pose tokens for classifier-free guidance unconditional branch."""
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype
        return torch.zeros(batch_size, self.seq_len, self.context_dim, device=device, dtype=dtype)
