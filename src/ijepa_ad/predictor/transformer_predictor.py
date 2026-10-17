"""Lightweight Transformer predictor over cached I-JEPA latent tokens."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class PredictorConfig:
    """Architecture fields for the latent Transformer predictor."""

    embedding_dim: int
    num_tokens: int
    hidden_dim: int
    depth: int
    num_heads: int
    mlp_ratio: float
    dropout: float

    def __post_init__(self) -> None:
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if self.num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.depth <= 0:
            raise ValueError("depth must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class LatentTransformerPredictor(nn.Module):
    """Predict masked latent tokens from visible latent context tokens."""

    def __init__(self, config: PredictorConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.embedding_dim, config.hidden_dim)
        self.output_projection = nn.Linear(config.hidden_dim, config.embedding_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.num_tokens, config.hidden_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=int(config.hidden_dim * config.mlp_ratio),
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.depth)
        self.norm = nn.LayerNorm(config.hidden_dim)
        self._reset_parameters()

    def forward(self, latents: Tensor, target_mask: Tensor) -> Tensor:
        """Return predictions for every token, with loss applied on ``target_mask``."""
        if latents.ndim != 3:
            raise ValueError(f"latents must be B x tokens x dim, got {latents.ndim}D")
        if latents.shape[1:] != (self.config.num_tokens, self.config.embedding_dim):
            raise ValueError(
                f"latents must have token/dim shape "
                f"{(self.config.num_tokens, self.config.embedding_dim)}, got "
                f"{tuple(latents.shape[1:])}"
            )
        if target_mask.shape != latents.shape[:2]:
            raise ValueError(
                f"target_mask must be B x tokens, got {tuple(target_mask.shape)}"
            )
        parameter_dtype = self.input_projection.weight.dtype
        if latents.dtype != parameter_dtype:
            raise TypeError(
                f"latents are {latents.dtype} but the predictor parameters are "
                f"{parameter_dtype}; cast the latents (e.g. cached float16) first"
            )
        mask = target_mask.to(device=latents.device, dtype=torch.bool)
        if not bool(mask.any()):
            raise ValueError("target_mask must select at least one token")

        encoded = self.input_projection(latents)
        encoded = torch.where(mask.unsqueeze(-1), self.mask_token, encoded)
        encoded = encoded + self.position_embedding
        predicted = self.encoder(encoded)
        return self.output_projection(self.norm(predicted))

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
